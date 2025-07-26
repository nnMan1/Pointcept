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
    def __init__(self, model_name="facebook/dinov2-small", device="cuda"):
        super().__init__()
        self.device = device

        # Load pretrained DINOv2 model
        self.model = Dinov2Model.from_pretrained(model_name).to(self.device).eval()

    def forward(self, data_dict):
        images = data_dict.get('images', [])
            
        mesh_features = torch.zeros((len(data_dict['coord']), 384), dtype=torch.float32, device=self.device)
        mesh_features_cnt = torch.zeros((len(data_dict['coord']), 384), dtype=torch.float32, device=self.device)

        bs, ibs, fbs, obs = 0, 0, 0, 0
        i=0

        for be, ibe, obe, in zip(data_dict['offset'], data_dict['image_offset'], data_dict['mappings_offset']):
            outputs = self.model(images[ibs: ibe], output_hidden_states=True)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # remove CLS token

            features = patch_tokens.to(self.device)
            
            mappings_src = data_dict['mappings_src'][obs:obe]
            mappings_tgt = data_dict['mappings_tgt'][obs:obe]

            print(features.shape)
            # for tokens, mapping in zip(patch_tokens[ibs:ibe], mappings[ibs:ibe]):
            #     n_patches = tokens.shape[0]
            #     dim = int(n_patches ** 0.5)
            #     assert dim * dim == n_patches, "Patch tokens are not square!"

            #     tokens_2d = tokens.reshape(dim, dim, -1).permute(2, 0, 1).unsqueeze(0)
            #     tokens_2d = F.interpolate(tokens_2d, size=(mapping.shape[0], mapping.shape[1]), mode='bilinear', align_corners=False)
            #     tokens_2d = tokens_2d.squeeze(0).permute(1, 2, 0)  # [H, W, C]

            #     mask = mapping > 0
            #     mesh_features[bs:be][faces[mapping[mask]]] += tokens_2d[mask].unsqueeze(1)
            #     mesh_features[bs:be][faces[mapping[mask]]] += 1

            # bs = be
            # ibs = ibe
            # obs = obe

        mesh_features[mesh_features_cnt > 0] = mesh_features[mesh_features_cnt > 0] / mesh_features_cnt[mesh_features_cnt > 0]

        return mesh_features

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
