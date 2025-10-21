import functools
import gorilla
import spconv.pytorch as spconv
import torch
from collections import OrderedDict
import os
from spconv.pytorch.modules import SparseModule
from torch import nn
import torch_scatter
from pointcept.models.utils.structure import Point
from typing import Callable, Dict, List, Optional, Union
from pointcept.models.utils import offset2batch
from pointcept.models.point_transformer_v3 import PointTransformerV3
from pointcept.utils.visualization import pca_features_visualization

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
            if add_features[-1] is not None:
                if point.pooling_inverse != {}:
                    add_features.append(torch_scatter.scatter_mean(add_features[-1],  point.pooling_inverse, dim=0))
                else:
                    add_features.append(add_features[-1])

        if add_features[-1] is not None:
            add_features.pop(-1)
            add_features.reverse()

        

        for k, layer in self.dec._modules.items():
            point = layer(point)

            if add_features[0] is not None:
                point.feat = point.feat + add_features[0][:, :point.feat.shape[1]]
                add_features.pop(0)

        debugging = os.environ.get("DEBUGING", "false").lower() == "backbone"
        if debugging:

            print("Visualizing PCA features...")

            pca_features_visualization(
                point.coord.detach().cpu().numpy(),
                point.feat.detach().cpu().numpy(),
                n_components=3,
                file_path=f"backbone_features_{data_dict['name'][0]}_{k}.ply"
            )
            
        return point