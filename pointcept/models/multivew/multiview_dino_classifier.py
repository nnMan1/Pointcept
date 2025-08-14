import torch
import numpy as np
import math
import warnings
import torch.nn as nn
import torch.nn.functional as F

from .renderer import CADRenderer, CameraConfig
from typing import List, Tuple

from torch.nn.functional import interpolate

from pointcept.models.losses import build_criteria
from pointcept.models.builder import MODELS, build_model



def _to_tensor_batch(imgs: list[np.ndarray], device: torch.device, size: int = 224) -> torch.Tensor:
    """Convert a list of HxWxC uint8 images to a (B,3,H,W) float tensor in [0,1] resized to size.
    Uses torch.ops (no torchvision hard dependency for basic resize)."""
    
    x = torch.from_numpy(np.stack(imgs, axis=0)).to(device=device)
    # (B,H,W,C) -> (B,C,H,W), float in [0,1]
    x = x.permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != size or x.shape[-2] != size:
        x = interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    return (x - mean) / std


def load_dinov2_backbone(model_name: str = "vit_small_patch14_dinov2", pretrained: bool = True):
    """Load a DINOv2 backbone. Tries timm first; falls back to torch.hub if available.
    Returns (model, feature_dim). The model outputs a (N, D) feature tensor.
    """
    model = None

    # Try timm
    try:
        import timm
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        # Infer feature dim
        if hasattr(model, "num_features"):
            feat_dim = model.num_features
        elif hasattr(model, "num_features_head"):
            feat_dim = model.num_features_head
        else:
            # Fallback: run a dummy forward to infer
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                feat_dim = model(dummy).shape[-1]
        return model, int(feat_dim)
    except Exception as e:
        warnings.warn(f"timm load failed: {e}. Falling back to torch.hub if available.")

    # Try torch.hub
    try:
        import torch
        hub_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # Wrap to return penultimate features consistently
        class HubWrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)
        model = HubWrap(hub_model)
        feat_dim = 384  # vits
        return model, feat_dim
    except Exception as e:
        raise ImportError(
            "Could not load DINOv2 backbone via timm or torch.hub. Install timm (pip install timm) "
            "or ensure torch.hub can access facebookresearch/dinov2.") from e


@MODELS.register_module("MultiViewDinoClassifier")
class MultiViewDinoClassifier(nn.Module):
    """Small NN that renders K views with CADRenderer and classifies via DINOv2 features.

    Pipeline:
      packed-batch dict -> render K views per mesh -> DINOv2 encoder -> mean-pool over views -> MLP head
    """
    def __init__(
        self,
        num_classes: int,
        num_views: int = 4,
        backbone_name: str = "vit_small_patch14_dinov2",
        image_size: int = 224,
        freeze_backbone: bool = True,
        criteria: List[dict] = None,
    ) -> None:
        super().__init__()
        self.renderer = CADRenderer(device=torch.device("cuda:0"))
        self.num_views = num_views
        self.image_size = image_size

        encoder, feat_dim = load_dinov2_backbone(backbone_name, pretrained=True)
        self.encoder = encoder
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = max(256, feat_dim // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

        self.criteria = build_criteria(criteria)

    @torch.no_grad()
    def _render_views(self, batch: dict) -> torch.Tensor:
        """Render K views per mesh -> (B, K, 3, H, W) float normalized for DINOv2.
        Rendering is non-differentiable by design for speed.
        """
        B = batch["offset"].numel()  # number of meshes
        # Distribute azimuths around the object
        azims = [i * (360.0 / self.num_views) for i in range(self.num_views)]
        elev = 20.0
        dist = 2.5

        view_tensors: List[torch.Tensor] = []
        for az in azims:
            cam = CameraConfig(type="persp", azim=float(az), elev=elev, dist=dist, fov=20.0)
            self.renderer.load_from_packed_batch(batch, normalize=True, unit_scale=1.0)
            imgs = self.renderer.render_many(cam=cam, transparent=False)
            x = _to_tensor_batch(imgs, device=self.renderer.device, size=self.image_size)
            x = _imagenet_norm(x)
            view_tensors.append(x)  # (B,3,H,W)
        # Stack to (K,B,3,H,W) -> (B,K,3,H,W)
        X = torch.stack(view_tensors, dim=0).permute(1, 0, 2, 3, 4).contiguous()

        return X

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns logits: (B, num_classes)."""
        device = self.renderer.device
        X = self._render_views(batch)  # (B,K,3,H,W)

        B, K = X.shape[0], X.shape[1]
        X = X.view(B * K, 3, self.image_size, self.image_size)
        feats = self.encoder(X)  # (B*K, D)
        D = feats.shape[-1]
        feats = feats.view(B, K, D).mean(dim=1)  # (B, D)
        cls_logits = self.head(feats)  # (B, C)

        if self.training:
            loss = self.criteria(cls_logits, batch["category"])
            return dict(loss=loss)
        elif "category" in batch.keys():
            loss = self.criteria(cls_logits, batch["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)

# if __name__ == "__main__":
    # Example usage
    # renderer = CADRenderer(device=torch.device("cuda:0"))
    # model = MultiViewDinoClassifier(renderer, num_classes=10, num_views=4)
    
    # # Dummy batch with 2 meshes
    # batch = {
    #     "offset": torch.tensor([0, 1]),
    #     "coord": [torch.randn(100, 3), torch.randn(100, 3)],
    #     "faces": [torch.randint(0, 100, (50, 3)), torch.randint(0, 100, (50, 3))],
    # }
    
    # logits = model(batch)
    # print(logits.shape)  # Should print: torch.Size([2, 10])