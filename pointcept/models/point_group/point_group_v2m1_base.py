"""
PointGroup for instance segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Chengyao Wang
Please cite our work if the code is helpful to you.
"""

from functools import partial
import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
from pointops import batch2offset

from pointgroup_ops import ballquery_batch_p, bfs_cluster
from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.losses import FocalLoss, DiceLoss

from pointcept.models.builder import MODELS, build_model

class FocalLoss(torch.nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """

        target = target.long()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss
    
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def build(config):
        return Attention(config['embedding_dim'],
                         config['num_heads'],
                         downsample_rate=config['downsample_rate'] if 'downsample_rate' in config.keys() else 1)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        n, c = x.shape
        x = x.reshape(n, num_heads, c // num_heads)
        # return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head
        return x

    def forward(self, q: Tensor, k: Tensor, v: Tensor, batch: Tensor) -> Tensor:
        # Input projections

        batch = batch.clone().detach()
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        k = k[batch]
        v = v[batch]

        # Attention
        _, _, c_per_head = q.shape
        attn = q * k  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.cat([torch.softmax(attn[batch == i], dim = 0) for i in torch.arange(0, batch[-1] + 1)])

        # Get output
        out = attn * v
        out = out.reshape(out.shape[0], -1)
        out = self.out_proj(out)

        return out

@MODELS.register_module("PG-v2m1")
class PointGroupV2(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=300,
        cluster_propose_points=100,
        cluster_min_points=50,
        voxel_size=0.02,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.backbone = build_model(backbone)

        self.attn1 = Attention(embedding_dim = backbone_out_channels,
                               num_heads=8)
        
        self.attn2 = Attention(embedding_dim = backbone_out_channels,
                               num_heads=8)

        self.seg_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 2),
        )
        # self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)

        self.mask_dice_criteria = DiceLoss()
        self.mask_foc_criteria = FocalLoss()

        # self.ce_criteria = torch.nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)

    def forward(self, data_dict):

        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        instance_centroid = data_dict["instance_centroid"]
        seed_ids = data_dict["seed_ids"]
        offset = data_dict["offset"]

        batch = offset2batch(offset)
        
        feat = self.backbone(data_dict)

        masks = []
        loss_dice = []
        loss_focal = []

        for ids in seed_ids.T:
            ids[1:] += offset[:-1]
            k = feat[ids]
            v = k

            q = feat
            q = q + nn.ReLU()(self.attn1(q, k, v, batch))
            q = q + nn.ReLU()(self.attn2(q, q, v, batch))

            mask = self.seg_head(q)

            gt = torch.zeros_like(instance, dtype=torch.int32)
            gt[instance == instance[batch.clone().detach()]] = 1

            loss_dice.append(self.mask_dice_criteria(mask, gt))
            loss_focal.append(self.mask_foc_criteria(mask, gt))

            masks.append(mask)
            
        loss_dice = torch.mean(loss_dice)
        loss_focal = torch.mean(loss_focal)
        loss = loss_focal + loss_dice
        
        return_dict = {
            'loss': loss,
            'loss_dice': loss_dice,
            'loss_dice': loss_dice,
        }

        return return_dict