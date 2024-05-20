import torch
from torch import nn
from torch.nn import functional as F
from pointcept.utils.visualization import nms


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        # x: n x c x hw -> hw x n x c
        x = x.permute(2, 0, 1)
        return x

NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(1, dtype=torch.int32)
    )
    return intersections / (unions + 1e-15)

def filter_out_masks(masks, stabilities, ious=None, offset=None):
    pred_masks = []
    pred_stabilities = []
    pred_ious_ = []

    batch_start = 0

    for i, batch_end in enumerate(offset):
        preds = masks[batch_start: batch_end]
        stability = stabilities[i]

        if ious is not None:
            iou = ious[i]

        st_tras = min(0.5, stability.max())

        filter = (stability >= st_tras) #& (pred_ious > 0.3)
        preds = preds[:, filter]
        stability = stability[filter]

        if ious is not None:
            iou = iou[filter]
        
        keep = nms(preds, stability, 0.3)
        preds = preds[:, keep]
        stability = stability[keep]

        if ious is not None:
            iou = iou[keep]

        pred_masks.append(preds)
        pred_stabilities.append(stability)

        if ious is not None:
            pred_ious_.append(iou.cpu())

        batch_start = batch_end

    return pred_masks, pred_stabilities

def pad_batch(data, offset, size = None, rand_idx = None, mask_idx = None):

    if rand_idx is None:
        rand_idx = []
        mask_idx = []

        batch_start = torch.cat([torch.tensor([0]), offset[:-1]])

        max_size = (offset - batch_start).max()

        if size != None:
            max_size = min(max_size, size)

        batch_start = 0

        for i, batch_end in enumerate(offset):
            pcd = data[batch_start: batch_end]
            pcd_size = batch_end - batch_start

            if pcd_size < max_size:
                idx = torch.zeros(max_size,
                                dtype=torch.long,
                                device=data.device)
                

                midx = torch.ones(max_size,
                                dtype=torch.bool,
                                device=data.device)
                
                idx[:pcd_size] = torch.arange(
                    pcd_size, device = data.device
                )

                midx[:pcd_size] = False

            else:
                idx = torch.randperm( pcd_size,
                                    device = data.device
                                    )[:max_size]
                midx = torch.zeros(max_size,
                                dtype=bool,
                                device=data.device)
                

            rand_idx.append(idx)
            mask_idx.append(midx)

            batch_start = batch_end

    batched_data = []
    
    batch_start = 0

    for i, batch_end in enumerate(offset):
        batched_data.append(data[batch_start:batch_end+1][rand_idx[i]])
        batch_start = batch_end
        
    batched_data = torch.stack(batched_data)
    
    return batched_data, rand_idx, mask_idx
            