from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformers import Dinov2Model
from torchvision import transforms

from  pointcept.models.builder import MODELS
from .base_feature_extractor import BaseFeatureExtractor

Tensor = torch.Tensor
Batch = Mapping[str, Tensor]
Out = Dict[str, Tensor]

@MODELS.register_module("DinoV2FeatureExtractor")
class DinoV2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, 
                 model_name="facebook/dinov2-small",
                 merge_strategy='mean', 
                 fts_dim=384,
                 out_fts_dim=256,
                 return_features=None, 
                 **kwargs,
                 ):
        
        super().__init__(return_features=return_features, **kwargs)
        self.merge_strategy = merge_strategy
        self.fts_dim = fts_dim
        self.model = Dinov2Model.from_pretrained(model_name, 
                                                 local_files_only=True)
        
        self.proj = nn.Linear(self.fts_dim, self.out_fts_dim)

    def backbone_modules(self):
        return [self.model]

    def forward_features(self, batch: Batch) -> Dict[str, Tensor]:

        images = batch.get('images', [])

        mesh_features = torch.zeros((len(batch['coord']), self.fts_dim), dtype=torch.float32, device=self.model.device)
        mesh_features_cnt = torch.zeros((len(batch['coord'])), dtype=torch.float32, device=self.model.device)

        bs, ibs, mbs, obs = 0, 0, 0, 0
        i=0

        for be, ibe, mbe, obe in zip(batch['offset'], batch['image_offset'], batch['mappings_offset'], batch['origin_offset']):
            outputs = self.model(images[ibs: ibe], output_hidden_states=True)

            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # remove CLS token

            n_patches = patch_tokens.shape[1]
            dim = int(n_patches ** 0.5)
            assert dim * dim == n_patches, "Patch tokens are not square!"

            img_size = torch.tensor(images[ibs].shape[-2:], dtype=torch.int32)
            div_factor = 512 // dim

            patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], dim, dim, -1) 

            features = patch_tokens.to(self.model.device)

            mappings_src = batch['mappings_src'][mbs:mbe]
            mappings_tgt = batch['mappings_tgt'][mbs:mbe]
            inverse = batch['inverse'][obs:obe]
            mappings_tgt = inverse[mappings_tgt] # Apply inverse mapping from points to voxels

            if self.merge_strategy == 'random_sample':
                random_positions = self.one_random_position_per_value(mappings_tgt)

                mappings_src = mappings_src[random_positions]
                mappings_tgt = mappings_tgt[random_positions]

            a, b, c = mappings_src.T
            b //= div_factor
            c //= div_factor

            mesh_features[bs:be][mappings_tgt] += features[a, b, c]
            mesh_features_cnt[bs:be][mappings_tgt] += 1

            bs = be
            ibs = ibe
            mbs = mbe
            obs = obe

        mesh_features[mesh_features_cnt > 0] = mesh_features[mesh_features_cnt > 0] / mesh_features_cnt[mesh_features_cnt > 0][..., None]
        mesh_features = self.proj(mesh_features)

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