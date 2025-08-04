import functools
import gorilla
import spconv.pytorch as spconv
import torch
from collections import OrderedDict
from spconv.pytorch.modules import SparseModule
from torch import nn
import torch_scatter
from pointcept.models.utils.structure import Point
from typing import Callable, Dict, List, Optional, Union
from pointcept.models.utils import offset2batch
from pointcept.models.point_transformer_v3 import PointTransformerV3

class PointTransformerV3AddFeatures(PointTransformerV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)

        add_features = [data_dict.get('add_features', None)]
        
        for k, layer in self.enc._modules.items():
            point = layer(point)
            if point.pooling_inverse != {}:
                add_features.append(torch_scatter.scatter_mean(add_features[-1],  point.pooling_inverse, dim=0))
            else:
                add_features.append(add_features[-1])

        add_features.pop(-1)
        add_features.reverse()

        for k, layer in self.dec._modules.items():
            point = layer(point)
            point.feat = point.feat + add_features[0][:, :point.feat.shape[1]]
            add_features.pop(0)

        return point