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
        mesh_features_cnt = torch.zeros((len(data_dict['coord'])), dtype=torch.float32, device=self.device)

        bs, ibs, mbs, obs = 0, 0, 0, 0
        i=0

        for be, ibe, mbe, obe in zip(data_dict['offset'], data_dict['image_offset'], data_dict['mappings_offset'], data_dict['origin_offset']):
            outputs = self.model(images[ibs: ibe], output_hidden_states=True)

            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # remove CLS token

            n_patches = patch_tokens.shape[1]
            dim = int(n_patches ** 0.5)
            assert dim * dim == n_patches, "Patch tokens are not square!"

            img_size = torch.tensor(images[ibs].shape[-2:], dtype=torch.int32)
            div_factor = img_size // dim

            patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], dim, dim, -1) # [B, C, H, W]
            # Save feature map of the first image as an image
            # if i == 0:
            #     # Take the first image's patch tokens and average over channels
            #     feature_map = patch_tokens[0].detach().cpu().numpy()  # shape: (dim, dim, feature_dim)
            #     feature_map_mean = feature_map.mean(axis=-1)  # shape: (dim, dim)
            #     feature_map_norm = (feature_map_mean - feature_map_mean.min()) / (feature_map_mean.ptp() + 1e-8)
            #     feature_img = (feature_map_norm * 255).astype(np.uint8)
            #     feature_img_pil = Image.fromarray(feature_img)
            #     feature_img_pil.save("first_image_features.png")

            features = patch_tokens.to(self.device)
            
            mappings_src = data_dict['mappings_src'][mbs:mbe]
            mappings_tgt = data_dict['mappings_tgt'][mbs:mbe]
            inverse = data_dict['inverse'][obs:obe]
            mappings_tgt = inverse[mappings_tgt] # Apply inverse mapping from points to voxels          

            mappings = torch.stack([mappings_src[:, 0], mappings_tgt])
            _, idx = np.unique(mappings.cpu(), axis=1, return_index=True)
            
            mappings_src = mappings_src[idx]
            mappings_tgt = mappings_tgt[idx]

            a, b, c = mappings_src.T
            b //= div_factor[0]
            c //= div_factor[1]

            mesh_features[bs:be][mappings_tgt] += features[a, b, c]
            mesh_features_cnt[bs:be][mappings_tgt] += 1

            bs = be
            ibs = ibe
            mbs = mbe
            obs = obe

        mesh_features[mesh_features_cnt > 0] = mesh_features[mesh_features_cnt > 0] / mesh_features_cnt[mesh_features_cnt > 0][..., None]

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
