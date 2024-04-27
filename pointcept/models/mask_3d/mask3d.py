
import torch
from torch import nn
import MinkowskiEngine as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
from pointcept.models.utils import offset2batch
from torch.cuda.amp import autocast
from torch.nn import functional as F

from pointcept.models.builder import MODELS, build_model
from .position_embedding import PositionEmbeddingCoordsSine
from pointcept.models.utils.matcher.hungarian_matcher import HungarianMatcher
from pointcept.models.utils.matcher.my_matcher import MyMatcher
from pointcept.models.losses import DiceLoss, FocalLoss, BinaryFocalLoss
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

import time

@MODELS.register_module("Mask-3D")
class Mask3D(nn.Module):
    
    def __init__(self, backbone, position_encoding, num_decoders, dim_feedforward, mask_dim, hidden_dim, mask_module_config, query_refinement_config):

        super().__init__()

        self.num_decoders = num_decoders
        self.mask_dim = mask_dim
        self.num_heads = 8
        self.dropout = 0.0
        self.hlevels = [0,1,2,3]

        self.backbone = build_model(backbone)        
        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        position_encoding.pop('type')
        self.pos_enc = PositionEmbeddingCoordsSine(**position_encoding)

        self.query_projection = GenericMLP(
                input_dim=mask_dim,
                hidden_dims=[mask_dim],
                output_dim=mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

        sample_sizes = [200, 800, 3200, 12800, 51200]
        sizes = self.backbone.PLANES[-5:]
       
        self.mask_module = MaskModule(hidden_dim, **mask_module_config, num_masks=3)
        self.query_refinement = nn.ModuleList()

        for i, hlevel in enumerate(self.hlevels):
            self.query_refinement.append(QueryRefinement(sizes[i], dim_feedforward, mask_dim, sample_size=sample_sizes[i], **query_refinement_config))

        # self.matcher = HungarianMatcher()
        self.matcher = MyMatcher()

        self.loss_ce = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_dice = DiceLoss()
        self.loss_focal = BinaryFocalLoss(reduce=None)

    def __get_pos_encs(self, coords):

        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        coords_batch[None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )

                if tmp.sum().isnan():
                    pass

                pos_encodings_pcd[-1].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward(self, data):
        
        raw_coordinates = data['coord']
        grid_coordinates = data['grid_coord']
        offset = data['offset']
        seed_ids = data['seed_ids']

        # for p in data:
        #     print(data['path'])
        
        mask_features, aux = self.backbone(data)
        pcd_features = aux[-1]
        
        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=data['grid_coord'].float(),
                coordinate_manager=aux[-1].coordinate_manager,
                coordinate_map_key=aux[-1].coordinate_map_key,
                device=aux[-1].device,
            )

            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

            

        pos_encodings_pcd = self.__get_pos_encs(coords)

        # if self.train_on_segments:
        #     mask_segments = []
        #     for i, mask_feature in enumerate(
        #         mask_features.decomposed_features
        #     ):
        #         mask_segments.append(
        #             self.scatter_fn(mask_feature, point2segment[i], dim=0)
        #         )

        sampled_coords = []
        sampled_features = []
        batch_start = 0
        for i, batch_end in enumerate(offset):
            points = grid_coordinates[batch_start:batch_end].float()
            features = mask_features.features[batch_start:batch_end]
            sampled_coords.append(points[seed_ids[i]])
            sampled_features.append(features[seed_ids[i]])

        mins = torch.stack(
                [
                    coordinates.decomposed_features[i].min(dim=0)[0]
                    for i in range(len(coordinates.decomposed_features))
                ]
            )
        maxs = torch.stack(
            [
                coordinates.decomposed_features[i].max(dim=0)[0]
                for i in range(len(coordinates.decomposed_features))
            ]
        )

        sampled_coords = torch.stack(sampled_coords)

        query_pos = self.pos_enc(
                sampled_coords.float(), input_range=[mins, maxs]
            )  # Batch, Dim, queries
        
        query_pos = self.query_projection(query_pos)

        queries = torch.zeros_like(query_pos).permute((0, 2, 1))
        # queries = torch.stack(sampled_features)

        query_pos = query_pos.permute((2, 0, 1))

        axiliary_losses = [], [], []

        for _ in range(1):
            for i in self.hlevels:
                masks = self.mask_module({
                    'query_feat': queries,
                    'query_pos': query_pos,
                    'mask_features': mask_features,
                    'num_pooling_steps': len(self.hlevels) - i 
                })

                matched_outputs, matched_targets, _ = self.matcher(masks, data, offset)
                # axiliary_losses.append(0)

                for mask, target in zip(matched_outputs, matched_targets):
                    axiliary_losses[0].append(self.loss_ce(mask, target).mean(1).min(0)[0].mean())
                    axiliary_losses[2].append(self.loss_focal(mask, target).mean(1).min(0)[0].mean())

                    # axiliary_losses[0]torch.stack(([self.loss_ce(m, target).mean(0) for m in mask])).min(0)[0].mean()
                    
                    # axiliary_losses[0].append(self.loss_ce(mask, F.one_hot(target, mask.shape[1]).float()))
                    # axiliary_losses[1].append(self.loss_dice(mask, target))
                    # axiliary_losses[2].append(self.loss_focal(mask, F.one_hot(target, mask.shape[1]).float()))

                # axiliary_losses[-1] /= len(matched_outputs)

                pos_encoding=torch.cat(pos_encodings_pcd[i])

                queries = self.query_refinement[i](
                                                   aux[i].features,
                                                   masks['attn_mask'].features,
                                                   pos_encoding,
                                                   torch.tensor([p[-1] for p in aux[i].decomposition_permutations]),
                                                   queries,
                                                   query_pos 
                                                   )
                
        masks = self.mask_module({
                    'query_feat': queries,
                    'query_pos': query_pos,
                    'mask_features': mask_features,
                    'num_pooling_steps': 0
                })
        
        matched_outputs, matched_targets, _ = self.matcher(masks, data, offset)
        intersections = []
        unions = []

        start_time = time.time()
        for mask, target in zip(matched_outputs, matched_targets):

            axiliary_losses[0].append(self.loss_ce(mask, target).mean(1).min(0)[0].mean())
            axiliary_losses[2].append(self.loss_focal(mask, target).mean(1).min(0)[0].mean())

            intersections.append(((mask > 0.5) * target).sum(1))
            unions.append(((mask > 0.5).sum(1) + target.sum(1)) - intersections[-1])
            
            ious = intersections[-1] / unions[-1]
            axiliary_losses[1].append(ious.max(0)[0].mean())
            
        intersections = torch.stack(intersections)
        unions = torch.stack(unions)
        
        ious = intersections / unions

        return_dict = {
            'bce_loss': torch.stack(axiliary_losses[0]).mean(),
            'focal_loss': torch.stack(axiliary_losses[2]).mean(),
            'iou': torch.stack(axiliary_losses[1]).mean()
        }
        
        return_dict['loss'] = return_dict['bce_loss'] + return_dict['focal_loss']

        if not self.training: 
            m = masks['outputs_masks'].clone()
            return_dict['pred_score'] = torch.zeros(len(offset), m.shape[0], m.shape[2])
            print(return_dict['pred_score'].shape)
            batch_start = 0
            for i, batch_end in enumerate(offset):
                m = masks['outputs_masks'][:, batch_start:batch_end]
                m = F.sigmoid(m)
                                
                return_dict['pred_score'][i] = (m * (m>0.5)).sum(1) / ((m>0.5).sum(1) + 1e-15)
                batch_start = batch_end
                
            return_dict['ious'] = ious.reshape(len(offset), -1, *ious[1:])
            return_dict['pred_score'] = return_dict['pred_score'].reshape(len(offset), -1, *ious[1:])

            return_dict['masks'] = masks['outputs_masks'].permute(1, 0, 2).reshape(masks['outputs_masks'].shape[1], -1)
            return_dict['matched_masks'] = matched_outputs
            return_dict['matched_targets'] = matched_targets

        torch.cuda.empty_cache()
        return return_dict
    
class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        # x: n x c x hw -> hw x n x c
        x = x.permute(2, 0, 1)
        return x

NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}
      
class MaskModule(nn.Module):

    def __init__(self, hidden_dim, num_classes, return_attn_masks, use_seg_masks=False, num_masks=1):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_masks = num_masks

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_masks * hidden_dim),
        )

        self.iou_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_masks * hidden_dim),
        )

        self.class_embed_head = nn.Linear(hidden_dim, num_classes)
        self.use_seg_masks = use_seg_masks

        self.pooling = MinkowskiAvgPooling(
            kernel_size=2, stride=2, dimension=3
        )

        self.return_attn_masks = return_attn_masks

    def forward(self, data):

        query_feat, mask_features, query_pos = data['query_feat'], data['mask_features'], data['query_pos']

        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []
        output_segments = []
        
        return_dict = {
            'outputs_class': outputs_class
        }

        if self.use_seg_masks:
            pass
            # point2segment = data['point2segment']
            # output_segments = []

            # for i in range(len(mask_segments)):
            #     output_segments.append(mask_segments[i] @ mask_embed[i].T)
            #     output_masks.append(output_segments[-1][point2segment[i]])
        
            # return_dict['output_segments'] = output_segments
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                embeddings = mask_embed[i]
                embeddings = embeddings.reshape(embeddings.shape[0], -1, self.num_masks).permute(2, 1, 0)

                output_masks.append(
                    torch.stack([mask_features.decomposed_features[i] @ embedding for embedding in embeddings]).permute(1, 0, 2)
                )

        output_masks = torch.cat(output_masks).permute(1, 0, 2)

        outputs_mask = [me.SparseTensor(
                features=m,
                coordinate_manager=mask_features.coordinate_manager,
                coordinate_map_key=mask_features.coordinate_map_key,
            ) for m in output_masks]

        return_dict['outputs_masks'] = output_masks

        if self.return_attn_masks:
            attn_mask = outputs_mask
            for _ in range(data['num_pooling_steps']):
                attn_mask = [self.pooling(a.float()) for a in attn_mask]

            a, _ = torch.stack([a.F.detach().sigmoid() < 0.5 for a in attn_mask]).max(0)

            attn_mask = me.SparseTensor(
                features=a,
                coordinate_manager=attn_mask[0].coordinate_manager,
                coordinate_map_key=attn_mask[0].coordinate_map_key,
            )
            
            return_dict['attn_mask'] = attn_mask
                        
        return return_dict

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
        
    def __pad(self, data, offset, size = None, rand_idx = None, mask_idx = None):

        if rand_idx is None:
            rand_idx = []
            mask_idx = []

            batch_start = torch.cat([torch.tensor([0]), offset[:-1]])

            max_size = (offset - batch_start).max()

            if size != None:
                max_size = min(max_size, size)

            batch_start = 0

            for i, batch_end in enumerate(offset):
                pcd = data[batch_start: batch_end]
                pcd_size = batch_end - batch_start

                if pcd_size < max_size:
                    idx = torch.zeros(max_size,
                                    dtype=torch.long,
                                    device=data.device)
                    

                    midx = torch.ones(max_size,
                                    dtype=torch.bool,
                                    device=data.device)
                    
                    idx[:pcd_size] = torch.arange(
                        pcd_size, device = data.device
                    )

                    midx[:pcd_size] = False

                else:
                    idx = torch.randperm( pcd_size,
                                        device = data.device
                                        )[:max_size]
                    midx = torch.zeros(max_size,
                                    dtype=bool,
                                    device=data.device)
                    

                rand_idx.append(idx)
                mask_idx.append(midx)

                batch_start = batch_end

        batched_data = []
        
        batch_start = 0

        for i, batch_end in enumerate(offset):
            batched_data.append(data[batch_start:batch_end+1][rand_idx[i]])
            batch_start = batch_end
            
        batched_data = torch.stack(batched_data)
        
        return batched_data, rand_idx, mask_idx
            
    def forward(self, point_features, attn_mask, pos_encoding, offset, queries, query_pos_encoding):

        point_features, rand_idx, mask_idx = self.__pad(point_features, offset, self.sample_size)
        attn_mask, _, _ = self.__pad(attn_mask, offset, self.sample_size, rand_idx, mask_idx)
        pos_encoding, _, _ = self.__pad(pos_encoding, offset, self.sample_size, rand_idx, mask_idx)


        attn_mask.permute((0, 2, 1))[
                    attn_mask.sum(1) == rand_idx[0].shape[0]
                ] = False
        
        m = torch.stack(mask_idx)
        attn_mask = torch.logical_or(attn_mask, m[..., None])

        src_pcd = self.lin_squeez(
                            point_features.permute((1, 0, 2))
                        )

        output = self.cross_attention(
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    # memory_mask=attn_mask.repeat_interleave(
                    #     self.num_heads, dim=0
                    # ).permute((0, 2, 1)),
                    # memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos_encoding.permute((1, 0, 2)),
                    query_pos=query_pos_encoding,
                )
        
        if output.isnan().sum() > 0:
            pass
        
        output = self.self_attention(
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos_encoding,
                )
        
        if output.isnan().sum() > 0:
            pass
        
        queries = self.ffn_attention(
                    output
                ).permute((1, 0, 2))
                
        return queries

class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        output = self.layers(x)
        return output
   
class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        
        if tgt.isnan().sum() > 0:
            pass

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]

        if tgt2.isnan().sum() > 0:
            pass
    
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )

class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        if tgt2.isnan().sum() > 0:
            pass
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )

class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

