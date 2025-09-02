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
                 **kwargs):
        super().__init__(freeze_backbone=freeze_backbone)
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

        return_dict = {}

        point.serialization(order=self.backbone.order, shuffle_orders=self.backbone.shuffle_orders)
        point.sparsify()

        point = self.backbone.embedding(point)

        add_features = [batch.get('add_features', None)]
        
        for k, layer in self.backbone.enc._modules.items():
            point = layer(point)
            if add_features[-1] is not None:
                if point.pooling_inverse != {}:
                    add_features.append(torch_scatter.scatter_mean(add_features[-1],  point.pooling_inverse, dim=0))
                else:
                    add_features.append(add_features[-1])

            return_dict[f'enc_{k}'] = point.feat

        if add_features[-1] is not None:
            add_features.pop(-1)
            add_features.reverse()        

        for k, layer in self.backbone.dec._modules.items():
            point = layer(point)

            if add_features[0] is not None:
                point.feat = point.feat + add_features[0][:, :point.feat.shape[1]]
                add_features.pop(0)

            return_dict[f'dec_{k}'] = point.feat

        return_dict['feat'] = point.feat

        return return_dict
