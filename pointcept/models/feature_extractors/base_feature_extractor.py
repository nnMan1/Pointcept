# packed_point_feature_extractor.py
from __future__ import annotations
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
Batch = Mapping[str, Tensor]
Out = Dict[str, Tensor]

# Optional acceleration (recommended)
try:
    from torch_scatter import scatter, segment_csr
    _HAS_SCATTER = True
except Exception:
    _HAS_SCATTER = False


# ------------------------------ Base (packed I/O) ------------------------------

class BaseFeatureExtractor(nn.Module):
    """
    Base for packed point-cloud feature extractors with dict I/O.

    Input batch (packed):
      - 'points': (P, 3)  float XYZ
      - 'feat'  : (P, Cin) optional per-point input channels
      - 'offset': (B,)    cumulative counts of points: e.g., [N1, N1+N2, ..., sum Ni]

      (Optionally, instead of 'offset', you may pass 'batch' as (P,) cloud indices)

    Output:
      - Named per-point features (P, Ck) for each name in `feature_names`
      - Optional 'global' (B, D) via segment pooling of the last feature

    Subclasses set:
      - self.feature_names: Sequence[str]
      - self.out_dims     : Dict[name, channels]
      - self.global_dim   : Optional[int]
    and implement:
      - forward_features(batch: Batch) -> Dict[str, Tensor]   # per-point packed tensors
    """

    def __init__(
        self,
        return_features: Optional[Sequence[str]] = None,
        *,
        global_pool: Optional[str] = "avg",   # 'avg' | 'max' | 'gem' | None
        normalize_global: bool = False,
        gem_p: float = 3.0,
        keep_keys: Optional[Iterable[str]] = None,  # passthrough keys
    ):
        super().__init__()
        if global_pool == "gem" and gem_p <= 0:
            raise ValueError("gem_p must be > 0 for GeM.")
        self.global_pool = global_pool
        self.normalize_global = normalize_global
        self.gem_p = float(gem_p)

        self.feature_names: Sequence[str] = []
        self.out_dims: Dict[str, int] = {}
        self.global_dim: Optional[int] = None

        self._return_features = tuple(return_features) if return_features is not None else None
        self._keep_keys = tuple(keep_keys) if keep_keys is not None else tuple()

    # -------- subclass hook --------
    def forward_features(self, batch: Batch) -> Dict[str, Tensor]:
        raise NotImplementedError

    # ------------- public API -------------
    def forward(self, batch: Batch) -> Out:
        self._validate_batch(batch)
        feats = self.forward_features(batch)  # each (P, Ck)
        for k, t in feats.items():
            assert t.dim() == 2, f"Feature '{k}' must be (P,C), got {tuple(t.shape)}."

        wanted = self._return_features or tuple(self.feature_names)
        out: Out = {k: feats[k] for k in wanted if k in feats}

        if self.global_dim is not None and self.global_pool is not None:
            last = self.feature_names[-1]
            if last not in feats:
                raise KeyError(f"Missing feature '{last}' for global pooling.")
            batch_idx, B = self._batch_index_and_B(batch)
            g = self._segment_pool(feats[last], batch_idx, B, self.global_pool, self.gem_p)  # (B,C)
            if self.normalize_global:
                g = F.normalize(g, p=2, dim=1, eps=1e-12)
            out["global"] = g

        for k in self._keep_keys:
            if k in batch:
                out[k] = batch[k]
        return out

    def set_return_features(self, names: Optional[Iterable[str]]) -> None:
        self._return_features = tuple(names) if names is not None else None

    def num_features(self) -> Mapping[str, int]:
        d = dict(self.out_dims)
        if self.global_dim is not None:
            d["global"] = self.global_dim
        return d

    @torch.no_grad()
    def feature_shapes(
        self,
        clouds: Sequence[int] = (1024, 1024),
        have_feat: bool = True,
        in_feat_dim: int = 3,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Tuple[int, ...]]:
        """
        Dummy pass to report output shapes with packed inputs.
        """
        dev = device or next(self.parameters()).device
        N = sum(clouds)
        pts = torch.randn(N, 3, device=dev)
        batch = {"points": pts, "offset": torch.tensor(self._lengths_to_offsets(clouds), device=dev)}
        if have_feat:
            batch["feat"] = torch.randn(N, in_feat_dim, device=dev)
        out = self.forward(batch)
        return {k: tuple(v.shape) for k, v in out.items()}

    # ------------- internals -------------
    @staticmethod
    def _validate_batch(batch: Batch) -> None:
        if "points" not in batch:
            raise KeyError("Batch must include 'points' (P,3).")
        pts = batch["points"]
        if pts.dim() != 2 or pts.size(-1) != 3:
            raise ValueError(f"'points' must be (P,3), got {tuple(pts.shape)}.")
        if ("offset" not in batch) and ("batch" not in batch):
            raise KeyError("Provide either 'offset' (B,) cumulative or 'batch' (P,) indices.")

    @staticmethod
    def _lengths_to_offsets(lengths: Sequence[int]) -> Sequence[int]:
        off = []
        s = 0
        for L in lengths:
            s += int(L)
            off.append(s)
        return off

    @staticmethod
    def _offsets_to_batch_index(offset: Tensor) -> Tuple[Tensor, int]:
        """
        offset: (B,) cumulative counts. Returns (batch_idx: (P,), B)
        """
        B = int(offset.numel())
        counts = torch.empty(B, dtype=torch.long, device=offset.device)
        counts[0] = offset[0]
        counts[1:] = offset[1:] - offset[:-1]
        P = int(offset[-1].item())
        idx = torch.repeat_interleave(torch.arange(B, device=offset.device), counts)
        assert idx.numel() == P
        return idx, B

    @staticmethod
    def _batch_index_and_B(batch: Batch) -> Tuple[Tensor, int]:
        if "batch" in batch:
            b = batch["batch"].long()
            B = int(b.max().item()) + 1 if b.numel() > 0 else 0
            return b, B
        else:
            return BasePackedFeatureExtractor._offsets_to_batch_index(batch["offset"].long())

    @staticmethod
    def _segment_pool(x: Tensor, batch_idx: Tensor, B: int, mode: str, p: float) -> Tensor:
        """
        x: (P,C), batch_idx: (P,), returns (B,C)
        """
        if B == 0:
            return x.new_zeros((0, x.size(1)))

        if _HAS_SCATTER:
            if mode == "avg":
                return scatter(x, batch_idx, dim=0, dim_size=B, reduce="mean")
            if mode == "max":
                return scatter(x, batch_idx, dim=0, dim_size=B, reduce="max")
            if mode == "gem":
                eps = 1e-6
                xp = x.clamp(min=eps).pow(p)
                return scatter(xp, batch_idx, dim=0, dim_size=B, reduce="mean").pow(1.0 / p)
            raise ValueError(f"Unknown pool mode: {mode}")

        # PyTorch fallback (no torch_scatter). Efficient enough for moderate B.
        C = x.size(1)
        device = x.device
        out = x.new_full((B, C), float("-inf") if mode == "max" else 0.0)
        if mode in ("avg", "gem"):
            if mode == "avg":
                sums = out.zero_()
                counts = torch.bincount(batch_idx, minlength=B).to(x.dtype).unsqueeze(1).clamp(min=1)
                sums.index_add_(0, batch_idx, x)
                return sums / counts
            else:  # GeM
                eps = 1e-6
                xp = x.clamp(min=eps).pow(p)
                sums = out.zero_()
                counts = torch.bincount(batch_idx, minlength=B).to(x.dtype).unsqueeze(1).clamp(min=1)
                sums.index_add_(0, batch_idx, xp)
                return (sums / counts).pow(1.0 / p)
        elif mode == "max":
            # loop over clouds (robust, simple)
            for b in range(B):
                mask = (batch_idx == b)
                if mask.any():
                    out[b] = x[mask].max(dim=0).values
                else:
                    out[b] = float("-inf")
            return out
        else:
            raise ValueError(f"Unknown pool mode: {mode}")
        
