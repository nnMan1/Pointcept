
import torch
import torch_scatter
from torch import nn

from .matchers import build_matcher
from ..builder import MODELS, build_model
from ..losses import build_criteria
from .nn import GenericMLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer, SuperpointPooling, SuperpointUnpooling, pad_data
from utils import *
from abc import ABC, abstractmethod

@MODELS.register_module("InstSegTransformerEncoder")
class InstSegTransformerEncoder(nn.Module):

    def __init__(self, backbone, out_channels, backbone_out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.backbone = build_model(backbone) 


    def forward(self, data):

        offset = data['offset']

        pcd_features = self.backbone(data)
        # mask_features = self.mask_features_head(pcd_features)
        
        return {
                'features': pcd_features, 
                'offset': offset
            }

@MODELS.register_module("InstSegTransformerDecoder")
class InstSegTransformerDecoder(nn.Module):

    def __init__(self, in_channels, mask_modules, query_refinement_modules):

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
                    
    def forward(self, point_features, attn_mask, offset, queries):

        point_features, rand_idx, mask_idx = pad_data(point_features, offset, self.sample_size)
        attn_mask, _, _ = pad_data(attn_mask, offset, self.sample_size, rand_idx, mask_idx)
                
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

@MODELS.register_module("InstanceSegmentationTransformerBase")
class InstanceSegmentationTransformerBase(nn.Module, ABC):
    def __init__(self,
                 encoder, 
                 decoder, 
                 matcher,
                 criteria_seg,
                 criteria_mask,
                 use_segments = False,
                 return_features = False
                ):

        super().__init__()

        self.use_segments = use_segments
        self.return_features = return_features

        self.encoder = build_model(encoder)
        self.decoder = build_model(decoder)

        self.matcher = build_matcher(matcher)
        self.criteria_seg = build_criteria(criteria_seg)
        self.criteria_mask = build_criteria(criteria_mask)
    
    def query_pooling(self, data):
        queries = self.__query.weight[None, ...].repeat(len(data['offset']), 1, 1) 
        
        return queries

    def __compute_loss(self, pred, data):
        
        axiliary_losses = {'seg_loss': torch.tensor(0.0),
                           'seg_loss': torch.tensor(0.0)}
                
        for p in pred:
            matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets, indices = self.matcher(p, data, data['offset'])
            matched_scores = [p['output_score'][i][indices[i][0]][...,0] for i in range(len(data['offset'])) if indices[i][0] is not None]

            t = {'seg_loss': torch.tensor(0.0),
                 'mask_loss': torch.tensor(0.0)}

            for score, mask, target, p_seg, t_seg in zip(matched_scores, matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets):
                t['seg_loss'] += self.criteria_seg(p_seg, t_seg, score)
                t['mask_loss'] += self.criteria_mask(mask, target)

            for key, value in axiliary_losses.items():
                axiliary_losses[key] += torch.stack(t[key]).mean() / len(matched_scores)

        for key, value in axiliary_losses.items():
            axiliary_losses[key] /= len(pred)
        
        axiliary_losses['loss'] = axiliary_losses['seg_loss'] + axiliary_losses['mask_loss']

        return axiliary_losses

    def forward(self, data):

        data.update(self.encoder(data))

        if self.use_segments:
            data = self.superpoint_pooling(data)

        queries = self.query_pooling(data)    
        pred = self.decoder(data, queries)  
        return_dict = self.__compute_loss(pred, data) 

        if self.return_features:
            return_dict['features'] = data['features']

        if not self.training:

            return_dict.update(select_masks(pred[-1], data['seg_indices'].cpu()))
            
            if self.use_segments:
                data = self.superpoint_unpooling(data)

        return return_dict
