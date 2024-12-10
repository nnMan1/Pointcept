
import torch
import torch_scatter
from torch import nn
from torch.cuda.amp import autocast

from pointcept.models.builder import MODELS, build_model
from pointcept.positional_embeddings import build_positional_embedding
from pointcept.models.utils.matcher.hungarian_matcher import HungarianMatcher
from pointcept.models.utils.matcher.my_matcher import MyMatcher
from pointcept.models.losses import DiceLoss, FocalLoss, BinaryFocalLoss
from .nn import GenericMLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer, pad_data
from .utils import compute_stats, select_masks

class Encoder(nn.Module):

    def __init__(self, backbone, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.backbone = build_model(backbone) 

        self.mask_features_head = nn.Linear(
            in_features=self.backbone.PLANES[7],
            out_features=self.out_channels,
            bias=True
        )

    def forward(self, data):

        offset = data['offset']
        seg_indices = data['seg_indices'] if 'seg_indices' in data.keys() else None

        pcd_features, aux = self.backbone(data)
        mask_features = self.mask_features_head(pcd_features)

        if seg_indices is not None: 

            return {
                'features': torch_scatter.scatter_mean(mask_features, seg_indices, dim=0), 
                'offset': torch.tensor([seg_indices[:off].max() for off in offset], dtype=torch.int), 
                'aux': aux
            }
        
        return {
                'features': mask_features, 
                'offset': offset, 
                'aux': aux
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

    def __init__query_features(self, query_points, query_embeddings):
        return torch.zeros_like(query_embeddings, device=query_embeddings.device)

    def forward(self, data, features, query_points):
        
        offset = data['offset']
        seg_indices = data['seg_indices'] if 'seg_indices' in data.keys() else None

        mask_features = features['features']

        out = []

        point_embedding = self.__get_pos_encs(features['aux'])

        query_embedding = self.__embed_queries(data, query_points)
        query_embedding = self.query_projection(query_embedding)
        query_features = self.__init__query_features(query_points, query_embedding).permute((0, 2, 1))

        attention_mask = torch.zeros([offset[-1], query_points.shape[1]], dtype=torch.bool, device=query_features.device)

        for mask_module in self.mask_modules:
            for _ in range(mask_module.reuse):

                for level in range(self.hlevels):

                    attention_mask = torch_scatter.scatter_mean(attention_mask.float(), features['aux'][level]['original_ids'], dim=0) > 0.5

                    pos_embedding = point_embedding[level]

                    query_features = self.query_refinement[level](
                                                    features['aux'][level]['features'],
                                                    attention_mask,
                                                    pos_embedding,
                                                    features['aux'][level]['offset'],
                                                    query_features,
                                                    query_embedding 
                                                    )

                    mask_module_data = {
                        'query_feat': query_features,
                        'query_pos': query_embedding,
                        'mask_features': mask_features,
                        'num_pooling_steps': self.hlevels - level,
                        'offset': offset
                    }

                    masks = mask_module(mask_module_data)

                    attention_mask = masks['attn_mask']

                    if seg_indices is not None:
                        attention_mask = attention_mask[seg_indices] # create per point attention mask

                    out.append(masks)
                
        return out

class MaskModule(nn.Module):

    def __init__(self, hidden_dim, num_classes, return_attn_masks, reuse=1):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.reuse = reuse

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
                
        self.class_embed_head = nn.Linear(hidden_dim, num_classes)

        self.return_attn_masks = return_attn_masks

    def forward(self, data):

        query_feat, mask_features, query_pos, offset = data['query_feat'], data['mask_features'], data['query_pos'], data['offset']
            
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []
        output_segments = []
        
        return_dict = {
            'outputs_class': outputs_class
        }

        bs = 0
        for i, be in enumerate(offset):
            output_masks.append(mask_features[bs:be] @ mask_embed[i].T)
            bs = be

        outputs_mask = torch.cat(output_masks)
        return_dict['outputs_mask'] = outputs_mask


        if self.return_attn_masks:
            attn_mask = outputs_mask.detach().sigmoid() < 0.5
            return_dict['attn_mask'] = attn_mask
                        
        return return_dict

class IoUHead(nn.Module):
    def __init__(self, in_channels, dim_feedforward, mask_dim, pre_norm, num_heads, dropout, sample_size):
        
        super().__init__()
        self.iou_transformer_head = QueryRefinement(in_channels, 
                                                    dim_feedforward=dim_feedforward, 
                                                    mask_dim=mask_dim, 
                                                    pre_norm=pre_norm, 
                                                    num_heads=num_heads, 
                                                    dropout=dropout, 
                                                    sample_size=sample_size)

        # self.iou_head = nn.Linear(in_channels, 1)
        self.iou_head = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.ious = None
    
        
    def forward(self, point_features, attn_mask, pos_encoding, offset, queries, query_pos_encoding):
        # return torch.zeros(queries.shape[:2], device = point_features.device)
        features = self.iou_transformer_head(point_features, attn_mask, pos_encoding, offset, queries, query_pos_encoding)
        features = queries

        shape = features.shape[:2]
        features = features.flatten(0, 1)
        ious = self.iou_head(features).squeeze(-1)
        ious = ious.reshape(*shape)
        return ious
  
class QueryRefinement(nn.Module):

    def __init__(self, in_channels, dim_feedforward, mask_dim, pre_norm, num_heads, dropout, sample_size):

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
                    
    def forward(self, point_features, attn_mask, pos_encoding, offset, queries, query_pos_encoding):

        point_features, rand_idx, mask_idx = pad_data(point_features, offset, self.sample_size)
        attn_mask, _, _ = pad_data(attn_mask, offset, self.sample_size, rand_idx, mask_idx)
        pos_encoding, _, _ = pad_data(pos_encoding, offset, self.sample_size, rand_idx, mask_idx)


        attn_mask.permute((0, 2, 1))[
                    attn_mask.sum(1) == rand_idx[0].shape[0]
                ] = False
        
        m = torch.stack(mask_idx)
        attn_mask = torch.logical_or(attn_mask, m[..., None])

        src_pcd = self.lin_squeez(
                            point_features
                        )
        
        attn_mask = attn_mask.permute((0, 2, 1))
        query_pos_encoding = query_pos_encoding.permute((0, 2, 1))
        
        output = self.cross_attention(
                    queries,
                    src_pcd,
                    memory_mask=None, #attn_mask.repeat_interleave(self.num_heads, dim=0),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos_encoding,
                    query_pos=query_pos_encoding,
                )
        
        output = self.self_attention(
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos_encoding,
                )
            
        queries = self.ffn_attention(
                    output
                )

        return queries

@MODELS.register_module("Mask-3D")
class Mask3D(nn.Module):
    
    def __init__(self, 
                 encoder, 
                 decoder,
                #  position_encoding, 
                #  num_decoders, 
                #  dim_feedforward, 
                #  mask_dim, 
                #  hidden_dim,
                 instance_ignore_index, 
                #  mask_module_config, 
                #  query_refinement_config
                ):

        super().__init__()

        self.instance_ignore_index = instance_ignore_index
        self.encoder = Encoder(**encoder)   

        for i, _ in enumerate(decoder['mask_modules']):
            decoder['mask_modules'][i]['num_classes'] += 1 #DUMMY CLASS FOR NONUSED PREDICTIONS

        self.decoder = Decoder(**decoder)
        self.matcher = HungarianMatcher(cost_class=2,
                                        cost_dice=2,
                                        cost_mask=5,
                                        instance_ignore_index=instance_ignore_index)
        
        self.loss_ce = nn.BCEWithLogitsLoss()
        weight = torch.ones(decoder['mask_modules'][0]['num_classes'])
        weight[-1] = 0.1

        self.loss_ce = nn.CrossEntropyLoss(weight=weight)
        self.loss_dice = DiceLoss()
        self.loss_focal = nn.BCEWithLogitsLoss()
        
        # self.iou_ce_loss = nn.BCEWithLogitsLoss()
        # self.iou_mse_loss = nn.MSELoss()

    def query_pooling(self, data):

        offset = data['offset']
        grid_coordinates = data['grid_coord']
        seed_ids = data['seed_ids']

        sampled_coords = []
        
        bs = 0
        for i, be in enumerate(offset):
            points = grid_coordinates[bs:be].float()
            sampled_coords.append(points[seed_ids[i]])
            bs = be

        return torch.stack(sampled_coords)

    def __prepare_seg_indices(self, seg_indices, offset):
        bb = 0
        off = 0

        offs = []

        for be in offset:
            tmp = seg_indices[bb:be]
            _, inverse_indices = torch.unique(tmp, return_inverse=True)
            seg_indices[bb:be] = inverse_indices + off
            off = seg_indices.max() + 1 
            offs.append(off)
            bb = be
    
        return seg_indices, torch.tensor(offs)

    def __compute_loss(self, pred, data):
        
        offset_key = 'offset'

        if 'seg_indices' in data.keys():
            seg_indices = data['seg_indices']
            instance = data['instance']
            segment = data['segment']

            # data['instance'] = torch_scatter.scatter_mean(torch.tensor(instance), torch.tensor(seg_indices))
            # data['segment'] =  torch_scatter.scatter_mean(torch.tensor(segment), torch.tensor(seg_indices))
            # offset_key = 'group_offset'


        axiliary_losses = {'seg_ce': [],
                           'mask_ce': [],
                           'mask_dice': [],
                           'matched_iou': []}

        for p in pred:
            
            p['outputs_mask'] = p['outputs_mask'][data['seg_indices']]
            matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets, indices = self.matcher(p, data, data['offset'])

            for mask, target, p_seg, t_seg in zip(matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets):
                axiliary_losses['seg_ce'].append(self.loss_ce(p_seg, t_seg))
                axiliary_losses['mask_ce'].append(self.loss_dice(mask, target))
                axiliary_losses['mask_dice'].append(self.loss_focal(mask, target.float()))
        
        intersections = []
        unions = []

        for mask, target, p_seg, t_seg in zip(matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets):
            intersections.append(((mask > 0.5) * target).sum(0))
            unions.append(((mask > 0.5).sum(0) + target.sum(0)) - intersections[-1])
            
            ious = intersections[-1] / unions[-1]
            axiliary_losses['matched_iou'].append(ious.mean())

        for key, value in axiliary_losses.items():
            axiliary_losses[key] = torch.stack(axiliary_losses[key]).mean()

        
        axiliary_losses['loss'] = 5 * axiliary_losses['mask_ce'] + 2 * axiliary_losses['mask_dice'] + axiliary_losses['seg_ce'] 
        
        return axiliary_losses

    def forward(self, data):

        if 'seg_indices' in data.keys():
            seg_indices = data['seg_indices']
            data['seg_indices'], data['group_offset'] = self.__prepare_seg_indices(data['seg_indices'], data['offset'])

        #TODO: URADI NESTO SA OVIM GOVNOM
        data['segment'][data['instance'] == self.instance_ignore_index] = -1
        # data['segment'] -= 2
        # data['group_segment'] -= 2
        # data['segment'][data['segment'] < 0] = -1
        # data['group_segment'][data['group_segment'] < 0] = -1
    
        features = self.encoder(data)
        queries = self.query_pooling(data)
        pred = self.decoder(data, features, queries)   

        return_dict = self.__compute_loss(pred, data)
        
        if not self.training:
            masks = pred[-1]
            # masks = db_scan(data, masks)
            return_dict.update(compute_stats(masks, data, data['group_offset']))

            return_dict['masks'] = masks['outputs_mask'][seg_indices.cpu()].T
            return_dict['pred_classes'] = masks['outputs_class'][..., :-1] #We remove dummy class from predictions
            
            return_dict['pred_masks'], return_dict['pred_scores'], return_dict['pred_classes'] = select_masks(return_dict['masks'].cpu(), return_dict['pred_classes'].cpu(), return_dict['pred_scores'].cpu(), offset=data['group_offset'])
            
            return_dict['pred_masks'] = return_dict['pred_masks'][0]
            return_dict['pred_scores'] = return_dict['pred_scores'][0]
            return_dict['pred_classes'] = return_dict['pred_classes'][0]
            # return_dict['matched_masks'] = matched_outputs
            # return_dict['matched_targets'] = matched_targets
            # return_dict['matched_seg_targets'] = matched_seg_targets


        return return_dict
