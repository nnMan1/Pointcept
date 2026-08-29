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
                 patch_size=None,
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

        # explicit override for models whose patch size differs from the
        # model_type default (e.g. future backbones); None keeps the default
        if patch_size is not None:
            self.div_factor = patch_size

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
                 proj_norm=False,
                 patch_size=None,
                 k_views=3,
                 **kwargs,
                 ):

        super().__init__(return_features=return_features, **kwargs)
        # pixel->patch divisor; None = derive from the feature grid at merge
        # time (works for any backbone), an int pins it explicitly
        self.patch_size = patch_size
        self.k_views = k_views  # for merge_strategy='ksample_mean'
        self.merge_strategy = merge_strategy
        self.fts_dim = fts_dim
        self.out_fts_dim = out_fts_dim
        self.project_fts = project_fts

        if model_type is not None:
            self.model = ImageFeatureExtractor(model_type, model_name, patch_size=patch_size)
        else:
            self.model = None

        if self.project_fts:
            proj_layers = [
                    nn.Linear(fts_dim, out_fts_dim),
                    nn.LayerNorm(out_fts_dim),
                    nn.ReLU(),
                    nn.Linear(out_fts_dim, out_fts_dim),
                ]
            if proj_norm:
                proj_layers.append(nn.LayerNorm(out_fts_dim))
            self.proj = nn.Sequential(*proj_layers)

    def backbone_modules(self):
        return [self.model]

    def forward_features(self, batch: Batch) -> Dict[str, Tensor]:

        images = batch.get('images', [])

        # Project-before-gather (random_sample only): proj is a pointwise MLP,
        # so gather(proj(grid)) == proj(gather(grid)). Projecting the compact
        # (V, H/p, W/p, C) grid runs the trainable head on ~20k vectors per
        # sample instead of one per mapped voxel (~5x less activation memory)
        # and lets the high-dim grid be freed right after each sample.
        # 'mean'/'max' must keep the original reduce-then-project order
        # (proj is nonlinear), so they use the legacy buffer below.
        # 'ksample_mean'/'center_weighted' aggregate PROJECTED features like
        # random_sample does (proj is applied to the compact grid first);
        # 'mean'/'max' keep the legacy reduce-then-project order.
        project_first = self.project_fts and self.merge_strategy in (
            'random_sample', 'ksample_mean', 'center_weighted')
        out_dim = self.out_fts_dim if project_first else self.fts_dim

        mesh_features = torch.zeros((len(batch['coord']), out_dim), dtype=torch.float32, device=self.device)
        mesh_features_cnt = torch.zeros((len(batch['coord'])), dtype=torch.float32, device=self.device)

        bs, ibs, mbs, obs = 0, 0, 0, 0

        for be, ibe, mbe, obe in zip(batch['offset'], batch['image_offset'], batch['mappings_offset'], batch['origin_offset']):
            if 'image_features' in batch:
                # precomputed (fp16) per-view patch grids, possibly kept in
                # pinned host memory (cfg.keep_on_cpu): stream only this
                # sample's views to GPU and upcast
                features = batch['image_features'][ibs:ibe].to(self.device, non_blocking=True).float()
            elif self.model is not None:
                with torch.no_grad():
                    features = self.model(images[ibs:ibe]).to(self.device)
            else:
                raise KeyError(
                    "Image2PointCLoud: batch has no 'image_features' and no image "
                    "backbone is configured (model_type=None)"
                )

            # Pixel->patch mapping derived from the actual grid: patch models
            # differ (DinoV3 512/16 -> 32x32, DinoV2 512/14 -> 36x36), so the
            # divisor cannot be hardcoded. (px * grid) // img, clamped, equals
            # px // 16 exactly for the 512/32 DinoV3 case (bit-identical), and
            # covers the full 36x36 grid for DinoV2 instead of reading only
            # the top-left 32x32 (the old hardcoded //16 bug).
            grid_h, grid_w = features.shape[1], features.shape[2]
            img_h, img_w = 512, 512

            def _to_patch(b, c):
                if self.patch_size is not None:
                    b = torch.clamp(b // self.patch_size, max=grid_h - 1)
                    c = torch.clamp(c // self.patch_size, max=grid_w - 1)
                else:
                    b = torch.clamp((b * grid_h) // img_h, max=grid_h - 1)
                    c = torch.clamp((c * grid_w) // img_w, max=grid_w - 1)
                return b, c

            if project_first:
                features = self.proj(features)

            with torch.no_grad():
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
                    b, c = _to_patch(b, c)

                    mesh_features[bs:be][mappings_tgt] += features[a, b, c]
                    mesh_features_cnt[bs:be][mappings_tgt] += 1
            elif self.merge_strategy == 'ksample_mean':
                # Average over k independent random view draws per voxel: keeps
                # random_sample's view-diversity regularization but cuts its
                # variance (k=1 is exactly random_sample, k=inf approaches mean).
                for _ in range(self.k_views):
                    rp = self.one_random_position_per_value(mappings_tgt)
                    if rp.numel() == 0:
                        continue
                    a, b, c = mappings_src[rp].T
                    b, c = _to_patch(b, c)
                    tgt = mappings_tgt[rp]
                    mesh_features[bs:be].index_add_(0, tgt, features[a, b, c])
                    mesh_features_cnt[bs:be].index_add_(
                        0, tgt, torch.ones_like(tgt, dtype=mesh_features_cnt.dtype))
            elif self.merge_strategy == 'center_weighted':
                # Weighted mean over ALL views seeing the voxel, weighting each
                # observation by how central it is in its image: peripheral
                # pixels are more distorted / grazing-angle, so their patch
                # features are less reliable than centered ones.
                a, b, c = mappings_src.T
                bq, cq = _to_patch(b, c)
                dy = (b.float() - img_h / 2.0) / (img_h / 2.0)
                dx = (c.float() - img_w / 2.0) / (img_w / 2.0)
                w = torch.clamp(1.0 - torch.sqrt(dy * dy + dx * dx) / 1.4142, min=0.05)
                mesh_features[bs:be].index_add_(0, mappings_tgt, features[a, bq, cq] * w[:, None])
                mesh_features_cnt[bs:be].index_add_(0, mappings_tgt, w.to(mesh_features_cnt.dtype))
            elif self.merge_strategy == 'mean':
                with torch.no_grad():
                    for i in range(ibe - ibs):
                        mask = (mappings_src[:, 0] == i)
                        if torch.sum(mask) == 0:
                            continue
                        selected_mappings_src = mappings_src[mask]
                        selected_mappings_tgt = mappings_tgt[mask]
                        a, b, c = selected_mappings_src.T
                        b, c = _to_patch(b, c)

                        mesh_features[bs:be][selected_mappings_tgt] += features[a, b, c]
                        mesh_features_cnt[bs:be][selected_mappings_tgt] += 1
            elif self.merge_strategy == 'max':
                with torch.no_grad():
                    for i in range(ibe - ibs):
                        mask = (mappings_src[:, 0] == i)
                        if torch.sum(mask) == 0:
                            continue
                        selected_mappings_src = mappings_src[mask]
                        selected_mappings_tgt = mappings_tgt[mask]
                        a, b, c = selected_mappings_src.T
                        b, c = _to_patch(b, c)

                        mesh_features[bs:be][selected_mappings_tgt] = torch.maximum(mesh_features[bs:be][selected_mappings_tgt], features[a, b, c])
                        mesh_features_cnt[bs:be][selected_mappings_tgt] = 1

            bs = be
            ibs = ibe
            mbs = mbe
            obs = obe

        if project_first:
            # random_sample writes exactly one (view, pixel) per voxel (counts
            # 0/1, nothing to average). ksample_mean/center_weighted accumulate
            # several (weighted) contributions, so they must be normalized.
            if self.merge_strategy != 'random_sample':
                with torch.no_grad():
                    valid = mesh_features_cnt > 0
                    mesh_features[valid] = mesh_features[valid] / mesh_features_cnt[valid][..., None]
        else:
            with torch.no_grad():
                valid = mesh_features_cnt > 0
                mesh_features[valid] = mesh_features[valid] / mesh_features_cnt[valid][..., None]

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
    

