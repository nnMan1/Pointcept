
import torch
import torch_scatter
from torch import nn
from torch.cuda.amp import autocast

from pointcept.models.builder import MODELS, build_model
from pointcept.positional_embeddings import build_positional_embedding
from pointcept.models.utils.matcher.hungarian_matcher import HungarianMatcher
from pointcept.models.utils.matcher.my_matcher import MyMatcher
from pointcept.models.losses import DiceLoss, FocalLoss, BinaryFocalLoss
from .nn import GenericMLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer, SuperpointPooling, SuperpointUnpooling, pad_data
from .utils import compute_stats, select_masks, db_scan

class Encoder(nn.Module):

    def __init__(self, backbone, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.backbone = build_model(backbone) 

        self.mask_features_head = nn.Sequential(
            nn.Linear(self.backbone.PLANES[7], out_channels), 
            nn.LayerNorm(out_channels), 
            nn.ReLU()
        )

    def forward(self, data):

        offset = data['offset']

        pcd_features = self.backbone(data)
        mask_features = self.mask_features_head(pcd_features)
        
        return {
                'features': mask_features, 
                'offset': offset
            }

class Decoder(nn.Module):

    def __init__(self, in_channels, positional_encoding, mask_modules, query_refinement_modules, hlevels):

        super().__init__()
        
        self.hlevels = hlevels

        self.query_projection = GenericMLP(
                input_dim=in_channels,
                hidden_dims=[in_channels],
                output_dim=in_channels,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

        self.pos_enc = build_positional_embedding(positional_encoding)

        self.mask_modules = nn.ModuleList([MaskModule(**cfg) for cfg in mask_modules])

        self.query_refinement = nn.ModuleList([QueryRefinement(**c) for c in query_refinement_modules])

    def __get_pos_encs(self, coords):

        pos_encodings_pcd = []

        for c in coords:
            pos_encodings_pcd.append([])

            bb = 0
            for be in c['offset']:
                scene_min = c['coords'][bb:be].min(dim=0)[0][None, ...]
                scene_max = c['coords'][bb:be].max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(c['coords'][bb:be][None, ...].float(), input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1].append(tmp.squeeze(0).permute((1, 0)))
                bb = be

            pos_encodings_pcd[-1] = torch.cat(pos_encodings_pcd[-1])

        return pos_encodings_pcd

    def __embed_queries(self, data, query_points):

        raw_coordinates = data['coord']
        offset = data['offset']

        mins, maxs = [], []

        bs = 0
        for i, be in enumerate(offset):
            coords = raw_coordinates[bs:be]

            mins.append(coords.min(dim=0)[0])
            maxs.append(coords.max(dim=0)[0])

            bs = be   

        mins = torch.stack(mins)
        maxs = torch.stack(maxs)
    
        query_pos = self.pos_enc(query_points.float(), input_range=[mins, maxs])

        return query_pos

    def forward(self, data, query_features):
        
        offset = data['offset']
        mask_features = data['features']

        out = []

        for mask_module in self.mask_modules:
            for _ in range(mask_module.reuse):

                for level in range(self.hlevels):

                    mask_module_data = {
                        'query_feat': query_features,
                        'mask_features': mask_features,
                        'offset': offset
                    }

                    masks = mask_module(mask_module_data)

                    attention_mask = masks['attn_mask']

                    query_features = self.query_refinement[0](
                                                    mask_features,
                                                    attention_mask,
                                                    data['offset'],
                                                    query_features,
                                                    )
                    
                    out.append(masks)

        mask_module_data = {
                        'query_feat': query_features,
                        'mask_features': mask_features,
                        'offset': offset
                    }

        masks = mask_module(mask_module_data)

        out.append(masks)
                
        return out

class MaskModule(nn.Module):

    def __init__(self, hidden_dim, num_classes, return_attn_masks, reuse=1):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.reuse = reuse

        self.decoder_norm = nn.LayerNorm(hidden_dim)                
        self.out_score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.class_embed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

        self.return_attn_masks = return_attn_masks

    def forward(self, data):

        query_feat, mask_features, offset = data['query_feat'], data['mask_features'], data['offset']
            
        query_feat = self.decoder_norm(query_feat)
        output_class = self.class_embed_head(query_feat)
        outputs_score = self.out_score(query_feat)

        output_masks = []
        output_segments = []
        
        return_dict = {
            'output_class': output_class,
            'output_score': outputs_score
        }

        bs = 0
        for i, be in enumerate(offset):
            output_masks.append(mask_features[bs:be] @ query_feat[i].T)
            bs = be

        outputs_mask = torch.cat(output_masks)
        return_dict['output_mask'] = outputs_mask

        if self.return_attn_masks:
            attn_mask = outputs_mask.detach().sigmoid() < 0.5
            return_dict['attn_mask'] = attn_mask
                        
        return return_dict

class QueryRefinement(nn.Module):

    def __init__(self, in_channels, dim_feedforward, mask_dim, pre_norm, num_heads, dropout, sample_size=None):

        super().__init__()

        self.dim_feed_forward = dim_feedforward
        self.mask_dim = mask_dim
        self.pre_norm = pre_norm
        self.num_heads = num_heads
        self.dropout = dropout
        self.sample_size = sample_size

        self.cross_attention = CrossAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
        
        self.lin_squeez =  nn.Linear(in_channels, self.mask_dim)
            
        self.self_attention = SelfAttentionLayer(
                    d_model=self.mask_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
        self.ffn_attention = FFNLayer(
                    d_model=self.mask_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=self.dropout,
                    normalize_before=self.pre_norm,
                )
                    
    def forward(self, point_features, attn_mask, offset, queries):

        point_features, rand_idx, mask_idx = pad_data(point_features, offset, self.sample_size)
        attn_mask, _, _ = pad_data(attn_mask, offset, self.sample_size, rand_idx, mask_idx)

        attn_mask.permute((0, 2, 1))[
                    attn_mask.sum(1) == rand_idx[0].shape[0]
                ] = False
                
        m = torch.stack(mask_idx)
        attn_mask = torch.logical_or(attn_mask, m[..., None])
        
        attn_mask = attn_mask.permute((0, 2, 1))
        
        output = self.cross_attention(
                    query = queries,
                    key = point_features,
                    value = point_features,
                    attn_mask=attn_mask.repeat_interleave(self.num_heads, dim=0),
                )
                
        output = self.self_attention(
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                )
                    
        queries = self.ffn_attention(
                    output
                )

        return queries

@MODELS.register_module("SPFormer")
class SPFormer(nn.Module):
    
    def __init__(self, 
                 num_query,
                 encoder, 
                 decoder,
                 instance_ignore_index, 
                ):

        super().__init__()

        self.instance_ignore_index = instance_ignore_index
        self.encoder = Encoder(**encoder)   

        self.superpoint_pooling = SuperpointPooling()
        self.superpoint_unpooling = SuperpointUnpooling()

        self.__query = nn.Embedding(num_query, encoder['out_channels'])

        for i, _ in enumerate(decoder['mask_modules']):
            decoder['mask_modules'][i]['num_classes'] += 1 #DUMMY CLASS FOR NONUSED PREDICTIONS

        self.decoder = Decoder(**decoder)
        self.matcher = HungarianMatcher(cost_class=.5,
                                        cost_dice=1,
                                        cost_mask=1,
                                        instance_ignore_index=instance_ignore_index)
        
        weight = torch.ones(decoder['mask_modules'][0]['num_classes'])
        weight[-1] = 0.1

        self.semantic_ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.mask_dice_loss = DiceLoss()
        self.mask_bce_loss = nn.BCEWithLogitsLoss()
        
        # self.iou_ce_loss = nn.BCEWithLogitsLoss()
        # self.iou_mse_loss = nn.MSELoss()
    
    def query_pooling(self, data):

        queries = self.__query.weight[None, ...].repeat(len(data['offset']), 1, 1) 
        
        return queries

    def __compute_loss(self, pred, data):
        
        axiliary_losses = {'seg_ce': [],
                           'mask_ce': [],
                           'mask_dice': [],
                           'matched_iou': [],
                           'score_loss': []}
        
        for p in pred:
            matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets, indices = self.matcher(p, data, data['offset'])

            for mask, target, p_seg, t_seg in zip(matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets):
                axiliary_losses['seg_ce'].append(self.semantic_ce_loss(p_seg, t_seg))
                axiliary_losses['mask_ce'].append(self.mask_bce_loss(mask, target.float()))
                axiliary_losses['mask_dice'].append(self.mask_dice_loss(mask, target))
                axiliary_losses['score_loss'].append(torch.nn.functional.mse_loss(mask.sigmoid(), target.float()))
        
        intersections = []
        unions = []

        for mask, target, p_seg, t_seg in zip(matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets):
            intersections.append(((mask > 0.5) * target).sum(0))
            unions.append(((mask > 0.5).sum(0) + target.sum(0)) - intersections[-1])
            
            ious = intersections[-1] / unions[-1]
            axiliary_losses['matched_iou'].append(ious.mean())

        for key, value in axiliary_losses.items():
            axiliary_losses[key] = torch.stack(axiliary_losses[key]).mean()
        
        axiliary_losses['loss'] = 0.5 * axiliary_losses['seg_ce'] + \
                                  1.0 * axiliary_losses['mask_ce'] + \
                                  1.0 * axiliary_losses['mask_dice']  + \
                                  0.5 * axiliary_losses['score_loss'] 
        return axiliary_losses

    def forward(self, data):

        data.update(self.encoder(data))

        data = self.superpoint_pooling(data)
        queries = self.query_pooling(data)    

        pred = self.decoder(data, queries)  

        return_dict = self.__compute_loss(pred, data)  


        if not self.training:
            masks = pred[-1]

            return_dict.update(compute_stats(masks, data, data['offset']))

            return_dict['pred_classes'] = masks['output_class'][..., :-1] #We remove dummy class from predictions

            return_dict['pred_masks'], return_dict['pred_scores'], return_dict['pred_classes'] = select_masks(masks['output_mask'].cpu(), return_dict['pred_classes'].cpu(), return_dict['pred_scores'].cpu(), offset=data['offset'])
            
            return_dict['pred_masks'] = return_dict['pred_masks'][0][data['seg_indices']].T
            return_dict['pred_scores'] = return_dict['pred_scores'][0]
            return_dict['pred_classes'] = return_dict['pred_classes'][0]

            ids = (return_dict['pred_masks'] > 0).sum(-1) >  100

            return_dict['pred_masks'] = return_dict['pred_masks'][ids]
            return_dict['pred_scores'] = return_dict['pred_scores'][ids]
            return_dict['pred_classes'] = return_dict['pred_classes'][ids]

            data = self.superpoint_unpooling(data)

        return return_dict
