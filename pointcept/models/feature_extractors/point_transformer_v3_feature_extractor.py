from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_scatter

from transformers import Dinov2Model
from torchvision import transforms

from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS
from pointcept.models import PointTransformerV3
from .base_feature_extractor import BaseFeatureExtractor

Tensor = torch.Tensor
Batch = Mapping[str, Tensor]
Out = Dict[str, Tensor]

@MODELS.register_module("PT-V3FeatureExtractor")
class PointTransformerV3FeatureExtractor(BaseFeatureExtractor):
    def __init__(self,
                 return_features: Optional[Sequence[str]] = None,
                 global_pool: Optional[str] = "None",   # 'avg' | 'max' | 'gem' | None
                 normalize_global: bool = False,
                 gem_p: float = 3.0,
                 keep_keys: Optional[Iterable[str]] = None,  # passthrough keys
                 freeze_backbone: bool = False,
                 freeze_backbone_bn: bool = False,
                 fusion: str = "average",  # 'average' | 'concat' for add_features
                 concat_norm: bool = False,  # LayerNorm each branch before concat
                 **kwargs):
        super().__init__(return_features=return_features,
                         global_pool=global_pool,
                         normalize_global=normalize_global,
                         gem_p=gem_p,
                         keep_keys=keep_keys,
                         freeze_backbone=freeze_backbone,
                         freeze_backbone_bn=freeze_backbone_bn)
        assert fusion in ("average", "concat"), f"unknown fusion '{fusion}'"
        self.fusion = fusion
        self.concat_norm = concat_norm
        self.backbone = PointTransformerV3(**kwargs)

        self.feature_groups = {
            "encoder": [f"enc_{i}" for i in range(len(self.backbone.enc))],
            "decoder": [f"dec_{i}" for i in range(len(self.backbone.dec))],
        }

        self.feature_groups["all"] = sum(self.feature_groups.values(), [])

    def backbone_modules(self):
        return [self.backbone]

    def forward_features(self, batch: Batch) -> Dict[str, Tensor]:
        point = Point(batch)

        fts_dist_loss = torch.tensor(0.0, device=point.feat.device)

        return_dict = {}

        point.serialization(order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders)
        point.sparsify()

        point = self.backbone.embedding(point)

        add_features = [batch.get('add_features', None)]
        
        for k, layer in self.backbone.enc._modules.items():
            point = layer(point)
            if add_features[-1] is not None:
                if point.pooling_inverse != {}:
                    add_features.append(torch_scatter.scatter_max(add_features[-1],  point.pooling_inverse, dim=0)[0])
                else:
                    add_features.append(add_features[-1])

            return_dict[f'enc_{k}'] = point.feat

        if add_features[-1] is not None:
            add_features.pop(-1)
            add_features.reverse()        

        for k, layer in self.backbone.dec._modules.items():
            point = layer(point)

            # 'average' fusion blends the (resolution-matched) DinoV3 features
            # into point.feat at every decoder level, truncated to the PT-V3
            # width, plus an MSE distillation term. 'concat' leaves the decoder
            # as pure PT-V3 and fuses once, after the loop, by concatenation.
            if self.fusion == "average" and add_features[0] is not None:
                valid_fts_mask = add_features[0].abs().sum(dim=1) > 0
                if valid_fts_mask.sum() > 0:
                    fts_dist_loss += F.mse_loss(point.feat[valid_fts_mask], add_features[0][valid_fts_mask, :point.feat.shape[1]].detach())
                    averaged = (point.feat + add_features[0][:, :point.feat.shape[1]]) / 2.0
                    new_feat = point.feat.clone()
                    new_feat[valid_fts_mask] = averaged[valid_fts_mask]
                    point.feat = new_feat
                    add_features.pop(0)

            return_dict[f'dec_{k}'] = point.feat

        # Late concat fusion: append the full-dim DinoV3 features to the final
        # decoder output (no truncation, no forced space alignment). The final
        # decoder output is at input resolution and row-aligned with the raw
        # add_features -- the same alignment the 'average' path uses at its last
        # decoder layer. Downstream backbone_out_channels must be widened by the
        # add_features dim (e.g. 64 + 256 = 320).
        if self.fusion == "concat" and batch.get('add_features', None) is not None:
            feat = point.feat
            dino = batch['add_features']
            # The DinoV3 projection ends in a bare Linear and PT-V3's feat is
            # also unnormalized, so the two branches reach the mask head at
            # uncontrolled (often very different) scales -- the larger one
            # dominates and slows the mask-feature head. Normalize each branch
            # so they contribute on equal footing.
            if self.concat_norm:
                feat = F.layer_norm(feat, feat.shape[-1:])
                dino = F.layer_norm(dino, dino.shape[-1:])
            point.feat = torch.cat([feat, dino], dim=-1)

        return_dict['feat'] = point.feat
        return_dict['loss'] = fts_dist_loss

        return return_dict
