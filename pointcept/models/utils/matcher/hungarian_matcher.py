# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
from multiprocessing import Pool
import torch_scatter
import numpy as np
from .builder import build_cost

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_terms: list,
        instance_ignore_index: int = -1
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_terms = nn.ModuleList()
        for ct in cost_terms:
            self.cost_terms.append(build_cost(ct))    

        self.instance_ignore_index = instance_ignore_index

    def my_optimized_forward(self, outputs, targets, offset):

        indices = []
        matched_outputs = []
        matched_targets = []
        matched_sem_outputs = []
        matched_sem_targets = []

        Cs = None

        for cost in self.cost_terms:
            if Cs is None:
                Cs = [cost.compute_cost(outputs, targets)]
            else:
                for o, n in zip(Cs, cost.compute_cost(outputs, targets)):
                    o += n

        Cs = [c.cpu() for c in Cs]
        for c in Cs:
            c = c.numpy()
            indices.append(linear_sum_assignment(c))

        for i, batch_end in enumerate(offset):
        
            pred_ids, tgt_ids,  = indices[i]
            out_mask = outputs['output_mask'][batch_start:batch_end]
            out_seg = outputs['output_class'][i]
            tgt_mask = targets['instance'][batch_start:batch_end]
            tgt_segm = targets['segment'][batch_start:batch_end]

            instances, idx = np.unique(tgt_mask.cpu(), return_index=True)
            if instances[0] == -1:
                idx = idx[1:]

            instances_seg = tgt_segm[idx]

            filter = tgt_mask != self.instance_ignore_index

            if filter.sum() == 0:
                batch_start = batch_end
                continue

            tgt_mask = F.one_hot(tgt_mask+1)[:, 1:]
            tgt_mask = tgt_mask[:, tgt_ids]

            out_mask = out_mask[:, pred_ids]

            tgt_segm = torch.ones_like(out_seg[:, 0], dtype=torch.int64) * (out_seg.shape[1] - 1)
            tgt_segm[pred_ids] = instances_seg[tgt_ids]
                       
            matched_outputs.append(out_mask)
            matched_targets.append(tgt_mask)
            matched_sem_outputs.append(out_seg)
            matched_sem_targets.append(tgt_segm)

            batch_start = batch_end
               
        return matched_outputs, matched_targets, matched_sem_outputs, matched_sem_targets, indices

    # @torch.no_grad()
    def forward(self, outputs, targets, offset):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.my_optimized_forward(outputs, targets, offset)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
