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
import time

# from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    ) / hw

    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    ) / hw

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        instance_ignore_index: int = -1
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.instance_ignore_index = instance_ignore_index

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points

        # self.p = Pool(128)

    # @torch.no_grad()
    def my_optimized_forward(self, outputs, targets, offset):

        start_time = time.time()
        batch_start = 0

        indices = []
        matched_outputs = []
        matched_targets = []
        matched_sem_outputs = []
        matched_sem_targets = []

        Cs = []
        for i, batch_end in enumerate(offset):

            with torch.no_grad():
                out_mask = outputs['outputs_mask'][batch_start:batch_end].T
                out_seg = outputs['outputs_class'][i].softmax(-1) 
                tgt_mask = targets['instance'][batch_start:batch_end]
                tgt_segm = targets['segment'][batch_start:batch_end]
                
                filter = tgt_mask != self.instance_ignore_index

                if filter.sum() == 0:
                    indices.append((None, None))
                    continue
                 
                tgt_mask = F.one_hot(tgt_mask+1).T[1:]
                instances_seg = (tgt_mask * tgt_segm).max(1)[0]
                cost_class = 1 - out_seg[:, instances_seg]

                tgt_mask = tgt_mask.float()

                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(
                    out_mask, tgt_mask
                )

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(
                    out_mask, tgt_mask
                )

                if torch.isinf(cost_mask).sum() > 0 or torch.isnan(cost_mask).sum() > 0:
                    pass
                    cost_mask = batch_sigmoid_ce_loss(
                        out_mask, tgt_mask
                    )

                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )


                C = C.cpu().numpy()
                indices.append(linear_sum_assignment(C))

            batch_start = batch_end

        batch_start = 0

        mapping_start = time.time()

        for i, batch_end in enumerate(offset):
        
            pred_ids, tgt_ids,  = indices[i]
            out_mask = outputs['outputs_mask'][batch_start:batch_end]
            out_seg = outputs['outputs_class'][i]
            tgt_mask = targets['instance'][batch_start:batch_end]
            tgt_segm = targets['segment'][batch_start:batch_end]

            # nonassigned_pred_ids = [i for i in range(out_mask.shape[-1]) if i not in pred_ids]
            

            filter = tgt_mask != self.instance_ignore_index

            if filter.sum() == 0:
                continue

            tgt_mask = F.one_hot(tgt_mask+1)[:, 1:]

            out_mask = out_mask[:, pred_ids]
            
            tmp = tgt_mask[:, tgt_ids].argmax(0)
            tmp = tgt_segm[tmp]

            tgt_segm = torch.zeros_like(out_seg[:, 0], dtype=torch.int64)
            tgt_segm[pred_ids] = tmp


            tgt_mask = tgt_mask[:, tgt_ids]#.argmax(-1)
                        
            matched_outputs.append(out_mask)
            matched_targets.append(tgt_mask)
            matched_sem_outputs.append(out_seg)
            matched_sem_targets.append(tgt_segm)

            batch_start = batch_end
            
        # print('Mapping time: ', time.time() - mapping_start)      
        # print('HM time: ', time.time() - start_time)      
        # print('Data_len: ', len(Cs))      
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
        # return self.memory_efficient_forward(outputs, targets, mask_type)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
