import torch
from torch import nn
from torch.nn import functional as F
from pointcept.utils.misc import batch_iou
from pointcept.utils.visualization import nms
from sklearn.cluster import DBSCAN
import numpy as np

    
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

def pad_data(data, offset, size = None, rand_idx = None, mask_idx = None):

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

def select_masks(masks, classes, stabilities, ious=None, offset=None):

        pred_masks = []
        pred_stabilities = []
        pred_ious_ = []

        if ious == None:
             ious = stabilities

        pred_ids = []

        batch_start = 0

        for i, batch_end in enumerate(offset):
            preds = masks[batch_start: batch_end]
            stability = stabilities[i]
            cls = classes[i]

            iou = ious[i]

            ids = torch.arange(len(stability))

            st_tras = 0.1

            filter = (stability > st_tras) #& (pred_ious > 0.3)
            preds = preds[:, filter]
            stability = stability[filter]
            iou = iou[filter]
            ids = ids[filter]
            
            keep = nms(preds, stability, 0.7).cpu()
            preds = preds[:, keep]
            stability = stability[keep]
            iou = iou[keep]
            ids = ids[keep]

            pred_masks.append(preds)
            pred_stabilities.append(stability)
            pred_ious_.append(iou.cpu())
            pred_ids.append(ids)

            batch_start = batch_end

        return pred_ids, pred_stabilities

def compute_stats(masks, data, offset, instance_ignore_index=-1):
        
        return_dict = {}
        
        m = masks['outputs_mask'].clone()
        return_dict['pred_scores'] = torch.zeros(len(offset), m.shape[1])
        return_dict['stability_score'] = torch.zeros(len(offset), m.shape[1])
        return_dict['bious'] =  torch.zeros(len(offset), m.shape[1], device='cuda')
        batch_start = 0
        
        for i, batch_end in enumerate(offset):
            m = masks['outputs_mask'][batch_start:batch_end]
            m = F.sigmoid(m)
            t = data['instance'][batch_start:batch_end]

            filter = t != instance_ignore_index
            filter = filter.cpu()

            m = m[filter]
            t = t[filter]

            if filter.sum() > 0:
                t = F.one_hot(t).float()
                biou = batch_iou((m.T > 0.5).float(), t.T)
                return_dict['bious'][i] = biou.max(-1)[0]
            else:
                return_dict['bious'][i] = torch.zeros(m.shape[1])

            return_dict['stability_score'][i] = calculate_stability_score(m, 0.5, 0.3)
            return_dict['pred_scores'][i] = masks['outputs_class'][i].softmax(-1)[..., 0:].max(-1)[0] * (m * (m>0.5)).sum(0) / ((m>0.5).sum(0) + 1e-15)
            
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

    clsses = preds['outputs_class'][0]
    masks = preds['outputs_mask'].T
    coords = data['coord'].cpu()

    for mask, cls in zip(masks, clsses):

        mask = mask.cpu()
        curr_masks = mask > 0
        curr_masks = curr_masks
        
        if coords[curr_masks].shape[0] > 0:
            clusters = (
                                DBSCAN(
                                    eps=0.95,
                                    min_samples=1,
                                    n_jobs=-1,
                                )
                                .fit(coords[curr_masks])
                                .labels_
                            )
            
            new_mask = torch.zeros(curr_masks.shape, dtype=int)
            new_mask[curr_masks] = (torch.from_numpy(clusters) + 1)
            

            for cluster_id in np.unique(clusters):
                if cluster_id != -1:
                    new_preds["pred_masks"].append(
                        mask * (new_mask == cluster_id + 1)
                    )
                    new_preds["pred_logits"].append(
                        cls
                    )

    print(len(new_preds['pred_logits']))
            
    preds['outputs_class'] = torch.stack(new_preds['pred_logits']).to(preds['outputs_class'].device)
    preds['outputs_mask'] = torch.stack(new_preds['pred_masks']).T.to(preds['outputs_class'].device)

    return preds