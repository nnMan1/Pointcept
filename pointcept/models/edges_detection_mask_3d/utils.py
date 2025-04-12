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

            batch_start = torch.cat([torch.tensor([0], device=offset.device), offset[:-1]])

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

def select_masks(masks, classes, offset=None):

        pred_masks = []
        pred_stabilities = []
        pred_ious_ = []
        ret_classes = []

        pred_ids = []

        batch_start = 0

        for i, batch_end in enumerate(offset):
            preds = masks[batch_start: batch_end]
            m = F.sigmoid(preds)
            cls = classes[i]

            scores = (cls.softmax(-1) * ((m * (m>0.5)).sum(0) / ((m>0.5).sum(0) + 1e-15))[:, None])[..., :-1]

            mask_id = torch.arange(len(scores))[..., None].repeat(1, scores.shape[-1])
            class_id = torch.arange(scores.shape[-1])[None, ...].repeat(len(scores), 1)

            scores, ids = scores.flatten().topk(50)
            mask_id = mask_id.flatten()[ids]
            class_id = class_id.flatten()[ids]

            preds = preds[:, mask_id]
            
            pred_masks.append(preds)
            pred_stabilities.append(scores)
            pred_ids.append(mask_id)
            ret_classes.append(class_id)

            batch_start = batch_end

        return pred_masks, pred_stabilities, ret_classes

def compute_stats(masks, data, offset, instance_ignore_index=-1):
        
        return_dict = {}
        
        m = masks['output_mask'].clone()
        return_dict['pred_scores'] = torch.zeros(len(offset), masks['output_class'].shape[1],  masks['output_class'].shape[2] - 1)
        return_dict['stability_score'] = torch.zeros(len(offset), m.shape[1])
        return_dict['bious'] =  torch.zeros(len(offset), m.shape[1], device='cuda')
        batch_start = 0
        
        for i, batch_end in enumerate(offset):
            m = masks['output_mask'][batch_start:batch_end].clone()
            m = F.sigmoid(m)
            t = data['instance'][batch_start:batch_end].clone()

            filter = t != instance_ignore_index
            filter = filter.cpu()

            if filter.sum() > 0:
                t = F.one_hot(t + 1).float()[:, 1:]
                biou = batch_iou((m.T > 0.5).float(), t.T)
                return_dict['bious'][i] = biou.max(-1)[0]
            else:
                return_dict['bious'][i] = torch.zeros(m.shape[1])

            return_dict['stability_score'][i] = calculate_stability_score(m, 0.5, 0.3)
            return_dict['pred_scores'][i] = (masks['output_class'][i].softmax(-1) * ((m * (m>0.5)).sum(0) / ((m>0.5).sum(0) + 1e-15))[:, None])[..., :-1]
            
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
            
    preds['outputs_class'] = torch.stack(new_preds['pred_logits']).to(preds['outputs_class'].device)[None, :]
    preds['outputs_mask'] = torch.stack(new_preds['pred_masks']).T.to(preds['outputs_class'].device)

    return preds