from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformers import Dinov2Model
from transformers import AutoModel, AutoImageProcessor, pipeline
from .base_feature_extractor import BaseFeatureExtractor

from  pointcept.models.builder import MODELS

Tensor = torch.Tensor
Batch = Mapping[str, Tensor]
Out = Dict[str, Tensor]

@MODELS.register_module("ImageFeatureExtractor")
class ImageFeatureExtractor(nn.Module):
    def __init__(self, 
                 model_type="DinoV2",
                 model_name="facebook/dinov2-small",
                ):
        super().__init__()
        self.model_type = model_type
        self.div_factor = 1

        if model_type == "DinoV2":
            self.model = Dinov2Model.from_pretrained(model_name, local_files_only=True)
            self.div_factor = 14
        elif model_type == "DinoV3":
            self.model = AutoModel.from_pretrained(model_name)
            self.div_factor = 16
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def __call__(self, images: Tensor) -> Tensor:

        outputs = self.model(images, output_hidden_states=True)

        if self.model_type == "DinoV2":
            patch_tokens = outputs.last_hidden_state[:, 1:, :] 
        elif self.model_type == "DinoV3":
            patch_tokens = outputs.last_hidden_state[:, 5:, :]  

        input_w, input_h = images.shape[-1], images.shape[-2]
        ogrid_w = input_w // self.div_factor
        ogrid_h = input_h // self.div_factor

        n_patches = patch_tokens.shape[1]
        dim = int(n_patches ** 0.5)
        assert dim * dim == n_patches, "Patch tokens are not square!"

        patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], ogrid_w, ogrid_h, -1) 

        return patch_tokens

    @property
    def device(self):
        return next(self.parameters()).device
    
@MODELS.register_module()
class Image2PointCLoud(BaseFeatureExtractor):
    def __init__(self, 
                 model_type = "DinoV2",
                 model_name = "facebook/dinov2-small",
                 merge_strategy='mean', 
                 fts_dim=384,
                 out_fts_dim=256,
                 return_features=None, 
                 local_files_only=False,
                 project_fts=True,
                 **kwargs,
                 ):
        
        super().__init__(return_features=return_features, **kwargs)
        self.merge_strategy = merge_strategy
        self.fts_dim = fts_dim
        self.out_fts_dim = out_fts_dim
        self.project_fts = project_fts

        if model_type is not None:
            self.model = ImageFeatureExtractor(model_type, model_name)
        else:
            self.model = None
        
        if self.project_fts:
            self.proj = nn.Sequential(
                    nn.Linear(fts_dim, out_fts_dim),
                    nn.LayerNorm(out_fts_dim),
                    nn.ReLU(),
                    nn.Linear(out_fts_dim, out_fts_dim)
                )

    def backbone_modules(self):
        return [self.model]

    def forward_features(self, batch: Batch) -> Dict[str, Tensor]:

        with torch.no_grad():
            images = batch.get('images', [])

            mesh_features = torch.zeros((len(batch['coord']), self.fts_dim), dtype=torch.float32, device=self.device)
            mesh_features_cnt = torch.zeros((len(batch['coord'])), dtype=torch.float32, device=self.device)

            bs, ibs, mbs, obs = 0, 0, 0, 0
            i=0

            for be, ibe, mbe, obe in zip(batch['offset'], batch['image_offset'], batch['mappings_offset'], batch['origin_offset']):
                if 'image_features' not in batch:
                    features = self.model(images[ibs:ibe])
                else:
                    features = batch['image_features']

                features = features.to(self.device)
                
                mappings_src = batch['mappings_src'][mbs:mbe]
                mappings_tgt = batch['mappings_tgt'][mbs:mbe]
                inverse = batch['inverse'][obs:obe]

                mappings_tgt = inverse[mappings_tgt] # Apply inverse mapping from points to voxels

                if self.merge_strategy == 'random_sample':
                    random_positions = self.one_random_position_per_value(mappings_tgt)

                    mappings_src = mappings_src[random_positions]
                    mappings_tgt = mappings_tgt[random_positions]

                    if mappings_tgt.numel() == 0:
                        print(f"[image-feat] skip empty mappings for sample={batch['name']}, bs:be={bs}:{be}")
                    else:
                        a, b, c = mappings_src.T
                        b //= 16
                        c //= 16

                        mesh_features[bs:be][mappings_tgt] += features[a, b, c]
                        mesh_features_cnt[bs:be][mappings_tgt] += 1
                elif self.merge_strategy == 'mean':
                    for i in range(ibe - ibs):
                        mask = (mappings_src[:, 0] == i)
                        if torch.sum(mask) == 0:
                            continue
                        selected_mappings_src = mappings_src[mask]
                        selected_mappings_tgt = mappings_tgt[mask]
                        a, b, c = selected_mappings_src.T

                        b //= 16
                        c //= 16

                        mesh_features[bs:be][selected_mappings_tgt] += features[a, b, c]
                        mesh_features_cnt[bs:be][selected_mappings_tgt] += 1
                elif self.merge_strategy == 'max':
                    for i in range(ibe - ibs):
                        mask = (mappings_src[:, 0] == i)
                        if torch.sum(mask) == 0:
                            continue
                        selected_mappings_src = mappings_src[mask]
                        selected_mappings_tgt = mappings_tgt[mask]
                        a, b, c = selected_mappings_src.T
                        b //= 16
                        c //= 16

                        mesh_features[bs:be][selected_mappings_tgt] = torch.maximum(mesh_features[bs:be][selected_mappings_tgt], features[a, b, c])
                        mesh_features_cnt[bs:be][selected_mappings_tgt] = 1


                bs = be
                ibs = ibe
                mbs = mbe
                obs = obe

            mesh_features[mesh_features_cnt > 0] = mesh_features[mesh_features_cnt > 0] / mesh_features_cnt[mesh_features_cnt > 0][..., None]

        if self.project_fts:
            mesh_features = self.proj(mesh_features)
            mesh_features[mesh_features_cnt == 0] = 0.0

        return {
            "feat": mesh_features
        }

    def one_random_position_per_value(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices : 1-D int/long tensor containing values (may repeat)
        returns : 1-D tensor of positions (in the input) —
                exactly one *random* position for every unique value.
        """
        perm = torch.randperm(indices.numel(), device=indices.device)
        shuffled_vals = indices[perm]

        _, first_in_shuffle = np.unique(shuffled_vals.cpu(), return_index=True)

        random_positions = perm[first_in_shuffle]

        return random_positions
    

