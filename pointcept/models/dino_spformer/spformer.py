
import os
import torch
import torch_scatter
from torch import nn
from torch.cuda.amp import autocast

from pointcept.models.builder import MODELS, build_model
from pointcept.positional_embeddings import build_positional_embedding
from pointcept.models.utils.matcher.hungarian_matcher import HungarianMatcher
from pointcept.models.losses import DiceLoss, FocalLoss, BinaryFocalLoss
from pointcept.models.utils.nn import GenericMLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer, SuperpointPooling, SuperpointUnpooling, pad_data
from .utils import compute_stats, select_masks, db_scan
from .backbone import PointTransformerV3AddFeatures
from pointcept.models.multivew.multiview_feaure_extraction import MeshFeatureExtractor
from pointcept.utils.visualization import pca_features_visualization

class Encoder(nn.Module):

    def __init__(self, 
                 backbone, 
                 out_channels, 
                 dino_version=None,
                 dino_output_size=384):
        super().__init__()
        self.out_channels = out_channels

        # self.backbone = build_model(backbone) 
        self.backbone = PointTransformerV3AddFeatures(**backbone)
        
        self.dino = None
        if dino_version is not None:
            self.dino = MeshFeatureExtractor(model_name=dino_version, merge_strategy="random_sample", fts_dim=dino_output_size)

            self.dino_mapping_mlp = nn.Sequential(
                nn.Linear(dino_output_size, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )

        self.mask_features_head = nn.Sequential(
            nn.Linear(64, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, data_dict):

        offset = data_dict['offset']

        if self.dino is not None:
            with torch.no_grad():
                data_dict['add_features'] = self.dino(data_dict)
                debugging = os.environ.get("DEBUGING", "false").lower() == "true"
                if debugging:
                    print("Visualizing DINO features...")
                    pca_features_visualization(
                        data_dict['coord'][:data_dict['offset'][0]].detach().cpu().numpy(),
                        data_dict['add_features'][:data_dict['offset'][0]].detach().cpu().numpy(),
                        n_components=3,
                        file_path=f"features_{data_dict['name']}_dino.ply"
                    )

            data_dict['add_features'] = self.dino_mapping_mlp(data_dict['add_features'])


        pcd_features = self.backbone(data_dict).feat
        mask_features = self.mask_features_head(pcd_features)
        
        return {
                'features': mask_features, 
                'offset': offset
            }

class Decoder(nn.Module):

    def __init__(self, in_channels, mask_modules, query_refinement_modules, hlevels):

        super().__init__()

        self.mask_features_head = nn.Sequential(nn.Linear(in_channels, mask_modules[0]['hidden_dim']), nn.ReLU(), nn.Linear(mask_modules[0]['hidden_dim'], mask_modules[0]['hidden_dim']))
        self.query_features_head = nn.Sequential(nn.Linear(in_channels, query_refinement_modules[0]['in_channels']), nn.LayerNorm(query_refinement_modules[0]['in_channels']), nn.ReLU())
        
        self.mask_modules = nn.ModuleList([MaskModule(**cfg) for cfg in mask_modules])
        self.query_refinements = nn.ModuleList([QueryRefinement(**c) for c in query_refinement_modules])

    def forward(self, data, query_features):
        
        offset = data['offset']
        features = data['features']

        mask_point_features = self.mask_features_head(features)
        query_point_features = self.query_features_head(features)

        out = []

        for mask_module in self.mask_modules:
            for _ in range(mask_module.reuse):
                for query_refinement in self.query_refinements:

                    mask_module_data = {
                        'query_feat': query_features,
                        'mask_features': mask_point_features,
                        'offset': offset
                    }

                    masks = mask_module(mask_module_data)

                    attention_mask = masks['attn_mask']

                    query_features = query_refinement(
                                                    query_point_features,
                                                    attention_mask,
                                                    data['offset'],
                                                    query_features,
                                                    pos = data['positional_embedding']
                                                    )
                    
                    out.append(masks)

        mask_module_data = {
                        'query_feat': query_features,
                        'mask_features': mask_point_features,
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
        attn_masks = []
        
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
            
            bs = 0
            for be in offset:
                attn_masks.append((outputs_mask[bs:be].sigmoid() < 0.5).bool())
                attn_masks[-1].permute(1, 0)[torch.where(attn_masks[-1].sum(0) == attn_masks[-1].shape[0])] = False
                bs = be

            return_dict['attn_mask'] = torch.cat(attn_masks).detach()

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
                    activation='gelu'
                )
                    
    def forward(self, point_features, attn_mask, offset, queries, pos):

        point_features, rand_idx, mask_idx = pad_data(point_features, offset, self.sample_size)
        attn_mask, _, _ = pad_data(attn_mask, offset, self.sample_size, rand_idx, mask_idx)

        if pos is not None:
            pos, _, _ = pad_data(pos, offset, self.sample_size, rand_idx, mask_idx)
                
        m = torch.stack(mask_idx)
        attn_mask = torch.logical_or(attn_mask, m[..., None])
        
        attn_mask = attn_mask.permute((0, 2, 1))
        
        output = self.cross_attention(
                    query = queries,
                    key = point_features,
                    value = point_features,
                    attn_mask=attn_mask.repeat_interleave(self.num_heads, dim=0),
                    pos=pos
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

@MODELS.register_module("MySPFormer")
class MySPFormer(nn.Module):
    
    def __init__(self, 
                 num_query,
                 encoder, 
                 decoder,
                 instance_ignore_index, 
                 positional_embedding = None
                ):

        super().__init__()

        self.instance_ignore_index = instance_ignore_index
        self.encoder = Encoder(**encoder)   

        self.superpoint_pooling = SuperpointPooling()
        self.superpoint_unpooling = SuperpointUnpooling()

        self.__query = nn.Embedding(num_query, decoder['query_refinement_modules'][0]['mask_dim'])

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

        self.positional_embedding = None
        
        if positional_embedding != None:
            self.positional_embedding = build_positional_embedding(positional_embedding)
    
    def query_pooling(self, data):
        queries = self.__query.weight[None, ...].repeat(len(data['offset']), 1, 1) 
        
        return queries

    def __compute_loss(self, pred, data):
        
        axiliary_losses = {'seg_ce': [],
                           'mask_ce': [],
                           'mask_dice': [],
                           'matched_iou': [],
                           'score_loss': []}
        

        intersections = []
        unions = []
        
        for p in pred:
            matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets, indices = self.matcher(p, data, data['offset'])
            matched_scores = [p['output_score'][i][indices[i][0]][...,0] for i in range(len(data['offset'])) if indices[i][0] is not None]

            t = {'seg_ce': [],
                 'mask_ce': [],
                 'mask_dice': [],
                 'matched_iou': [],
                 'score_loss': []}
            
            if len(matched_outputs) == 0:
                pass

            for score, mask, target, p_seg, t_seg in zip(matched_scores, matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets):
                t['seg_ce'].append(self.semantic_ce_loss(p_seg, t_seg))
                t['mask_ce'].append(self.mask_bce_loss(mask, target.float()))
                t['mask_dice'].append(self.mask_dice_loss(mask, target))

                with torch.no_grad():
                    intersections.append(((mask > 0) * target).sum(0))
                    unions.append(((mask > 0).sum(0) + target.sum(0)) - intersections[-1])
                    
                    ious = intersections[-1] / unions[-1]
                    t['matched_iou'].append(ious.mean())

                filter = ious > 0.5

                if filter.sum() > 0:
                    t['score_loss'].append(torch.nn.functional.mse_loss(score[filter], ious[filter]))
                else:  
                    t['score_loss'].append(torch.tensor(0.0).to(score.device))

            for key, value in axiliary_losses.items():
                if key in ['mask_dice']:
                    if len(t[key][:-1]) > 0:
                        axiliary_losses[key].append(torch.stack(t[key][:-1]).mean() + t[key][-1])
                    else:
                        axiliary_losses[key].append(t[key][-1])
                else:
                    axiliary_losses[key].append(torch.stack(t[key]).mean())

        for key, value in axiliary_losses.items():
            if key in ['matched_iou']:
                axiliary_losses[key] = torch.stack(axiliary_losses[key]).mean()
            else:
                axiliary_losses[key] = torch.stack(axiliary_losses[key]).sum()
        
        axiliary_losses['loss'] = 0.5 * axiliary_losses['seg_ce'] + \
                                  1.0 * axiliary_losses['mask_ce'] + \
                                  1.0 * axiliary_losses['mask_dice']  + \
                                  0.5 * axiliary_losses['score_loss']
        return axiliary_losses

    def __get_pos_encs(self, data_dict):

        if self.positional_embedding == None:
            data_dict['positional_embedding'] = None
            return data_dict

        pos_encodings_pcd = []

        bs = 0
        for be in data_dict['offset']:
            coords = data_dict['coord'][bs:be]
            scene_min = coords.min(dim=0)[0][None, ...]
            scene_max = coords.max(dim=0)[0][None, ...]

            with autocast(enabled=False):
                tmp = self.positional_embedding(coords[None, ...].float(), input_range=[scene_min, scene_max])

            pos_encodings_pcd.append(tmp.squeeze(0).permute((1, 0)))

        data_dict['positional_embedding'] = torch.cat(pos_encodings_pcd)

        return data_dict

    def forward(self, data):

        data.update(self.encoder(data))
        data.update(self.__get_pos_encs(data))

        data = self.superpoint_pooling(data, ['instance', 'segment', 'features'])

        queries = self.query_pooling(data)    

        pred = self.decoder(data, queries) 
        # pred = self.decoder(data['features'], data['offset']) 

        return_dict = self.__compute_loss(pred, data)  

        if not self.training:
            return_dict.update(select_masks(pred[-1], data['seg_indices'].cpu()))
            
            data = self.superpoint_unpooling(data)

        return return_dict
