import torch
from torch import nn
from torch.nn import functional as F
from pointcept.utils.misc import batch_iou
from sklearn.cluster import DBSCAN
import numpy as np


def mask_nms(masks, scores, iou_threshold):
    """NMS on binary masks using matrix IoU.
    Args:
        masks: (N, P) binary int/float tensor
        scores: (N,) float tensor
        iou_threshold: float
    Returns:
        keep indices into original ordering
    """
    n = masks.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=masks.device)

    order = torch.argsort(-scores)
    masks_float = masks[order].float()

    # Pairwise IoU via matrix multiply
    inter = masks_float @ masks_float.T  # (N, N)
    areas = masks_float.sum(dim=1)       # (N,)
    union = areas[:, None] + areas[None, :] - inter
    ious = inter / (union + 1e-6)

    keep = torch.ones(n, dtype=torch.bool, device=masks.device)
    for i in range(n):
        if not keep[i]:
            continue
        suppress = ious[i, i + 1:] > iou_threshold
        keep[i + 1:][suppress] = False

    return order[keep]

def select_masks(out, superpoints, score_thr=0.0, n_point_thr=100, topk=100, nms_thr=0.5):
        pred_labels = out['pred_logits'][0]
        mask_logits = out['pred_masks'].T
        pred_scores = out['pred_score'][0]

        num_class = pred_labels.shape[1] - 1
        num_query = pred_labels.shape[0]
        
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores
        labels = torch.arange(num_class, device=scores.device).unsqueeze(0).repeat(num_query, 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(topk, sorted=False)

        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, num_class, rounding_mode='floor')
        topk_masks = mask_logits[topk_idx]
        topk_masks_sigmoid = topk_masks.sigmoid()
        binary_masks = (topk_masks > 0).float()  # [n_p, M]
        mask_scores = (topk_masks_sigmoid * binary_masks).sum(1) / (binary_masks.sum(1) + 1e-6)
        scores = scores * mask_scores
        # expand to full points via superpoints
        full_masks = binary_masks[:, superpoints].int()

        # score_thr
        score_mask = scores > score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        full_masks = full_masks[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = full_masks.sum(1)
        npoint_mask = mask_pointnum > n_point_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        full_masks = full_masks[npoint_mask]  # (n_p, N)

        # early return if nothing survives filtering
        if full_masks.shape[0] == 0:
            return dict(
                pred_masks=np.empty((0, full_masks.shape[1]), dtype=np.int32),
                pred_scores=np.empty(0, dtype=np.float32),
                pred_classes=np.empty(0, dtype=np.int64),
            )

        nms_idx = mask_nms(full_masks, scores, nms_thr)
        scores = scores[nms_idx]  # (n_p,)
        labels = labels[nms_idx]  # (n_p,)
        full_masks = full_masks[nms_idx]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = full_masks.cpu().numpy()

        return dict(
            pred_masks=mask_pred,
            pred_scores=score_pred,
            pred_classes=cls_pred,
        )

def compute_stats(masks, data, offset, instance_ignore_index=-1):
        
        return_dict = {}
        
        return_dict['pred_scores'] = torch.zeros(len(offset), masks['output_class'].shape[1],  masks['output_class'].shape[2] - 1)
        return_dict['stability_score'] = torch.zeros(len(offset), masks['output_mask'].shape[1])
        return_dict['bious'] =  torch.zeros(len(offset), masks['output_mask'].shape[1], device='cuda')
        batch_start = 0
        
        for i, batch_end in enumerate(offset):
            m = masks['output_mask'][batch_start:batch_end].clone()
            m = F.sigmoid(m)
            t = data['instance'][batch_start:batch_end].clone()

            valid = t != instance_ignore_index

            if valid.sum() > 0:
                m_filtered = m[valid]
                t_filtered = t[valid]
                t_filtered = F.one_hot(t_filtered + 1).float()[:, 1:]
                biou = batch_iou((m_filtered.T > 0.5).float(), t_filtered.T)
                return_dict['bious'][i] = biou.max(-1)[0]
            else:
                return_dict['bious'][i] = torch.zeros(m.shape[1], device='cuda')

            return_dict['stability_score'][i] = calculate_stability_score(m, 0.5, 0.3)
            return_dict['pred_scores'][i] = (masks['output_class'][i].softmax(-1) * masks['output_score'][i])[..., :-1]
            batch_start = batch_end

        return return_dict
   
def calculate_stability_score(masks: torch.Tensor, mask_threshold: float, threshold_offset: float) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(0, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(0, dtype=torch.int32)
    )
    return intersections / (unions + 1e-15)

def db_scan(data, preds):

    new_preds = {
                        "pred_masks": list(),
                        "pred_logits": list(),
                    }

    clsses = preds['output_class'][0]
    masks = preds['output_mask'].T
    coords = data['coord']

    for mask, cls in zip(masks, clsses):

        curr_masks = mask > 0
        
        if coords[curr_masks].shape[0] > 0:
            clusters = (
                                DBSCAN(
                                    eps=0.95,
                                    min_samples=1,
                                    n_jobs=-1,
                                )
                                .fit(coords[curr_masks].cpu())
                                .labels_
                            )
            
            new_mask = torch.zeros(curr_masks.shape, dtype=int, device=curr_masks.device)
            new_mask[curr_masks] = (torch.from_numpy(clusters).to(curr_masks.device) + 1)
            

            for cluster_id in np.unique(clusters):
                if cluster_id != -1:
                    new_preds["pred_masks"].append(
                        mask * (new_mask == cluster_id + 1)
                    )
                    new_preds["pred_logits"].append(
                        cls
                    )

    if len(new_preds['pred_masks']) == 0:
        return preds

    preds['output_class'] = torch.stack(new_preds['pred_logits']).to(preds['output_class'].device)[None, :]
    preds['output_mask'] = torch.stack(new_preds['pred_masks']).T.to(preds['output_class'].device)

    return preds