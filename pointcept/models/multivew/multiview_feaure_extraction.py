from math import perm
import os
import torch
import glob
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from transformers import Dinov2Model
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class MeshFeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small", merge_strategy='mean', fts_dim=384):
        super().__init__()
        # self.device = device
        self.merge_strategy = merge_strategy

        # Load pretrained DINOv2 model
        print(model_name)
        self.model = Dinov2Model.from_pretrained(model_name, 
                                                 local_files_only=True).eval()

        self.fts_dim = fts_dim

    def forward(self, data_dict):
        images = data_dict.get('images', [])

        mesh_features = torch.zeros((len(data_dict['coord']), self.fts_dim), dtype=torch.float32, device=self.model.device)
        mesh_features_cnt = torch.zeros((len(data_dict['coord'])), dtype=torch.float32, device=self.model.device)

        bs, ibs, mbs, obs = 0, 0, 0, 0
        i=0

        for be, ibe, mbe, obe in zip(data_dict['offset'], data_dict['image_offset'], data_dict['mappings_offset'], data_dict['origin_offset']):
            outputs = self.model(images[ibs: ibe], output_hidden_states=True)

            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # remove CLS token

            n_patches = patch_tokens.shape[1]
            dim = int(n_patches ** 0.5)
            assert dim * dim == n_patches, "Patch tokens are not square!"

            img_size = torch.tensor(images[ibs].shape[-2:], dtype=torch.int32)
            div_factor = 512 // dim

            patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], dim, dim, -1) 

            features = patch_tokens.to(self.model.device)
            
            mappings_src = data_dict['mappings_src'][mbs:mbe]
            mappings_tgt = data_dict['mappings_tgt'][mbs:mbe]
            inverse = data_dict['inverse'][obs:obe]
            mappings_tgt = inverse[mappings_tgt] # Apply inverse mapping from points to voxels          

            mappings = torch.stack([mappings_src[:, 0], mappings_tgt])
            # _, idx = np.unique(mappings.cpu(), axis=1, return_index=True)
            
            # mappings_src = mappings_src[idx]
            # mappings_tgt = mappings_tgt[idx]

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

        return mesh_features

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

    def apply_pca_and_export(self, output_path='renders/example_colorized.ply', n_components=3):
        # Convert to CPU for PCA
        features = self.mesh_features.cpu().numpy()

        pca = PCA(n_components=n_components)
        tokens_pca = pca.fit_transform(features)

        # Normalize PCA to [0, 255]
        colors = tokens_pca[:, -3:]
        colors -= colors.min(axis=0)
        colors /= colors.max(axis=0)
        colors = (colors * 255).astype(np.uint8)

        self.mesh.visual.vertex_colors = colors
        self.mesh.export(output_path)
        print(f"Exported colorized mesh to {output_path}")
