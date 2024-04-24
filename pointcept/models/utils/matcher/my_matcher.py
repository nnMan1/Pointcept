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
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class MyMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self
    ):
        super().__init__()
    
    def my_optimized_forward(self, ouptups, targets, offset):

        batch_start = 0

        indices = []
        matched_outputs = []
        matched_targets = []

        for i, batch_end in enumerate(offset):

            out_mask = [o[batch_start:batch_end] for o in ouptups['outputs_masks']]
            tgt_mask = targets['instance'][batch_start:batch_end]
            tgt_mask = F.one_hot(tgt_mask).T
            
            seed_ids = targets['seed_ids'][i]
            seed_cls = targets['instance'][batch_start:batch_end][seed_ids]
            masks_tgt = tgt_mask[targets['instance'][batch_start:batch_end][seed_ids]].float()
                           
            matched_outputs.append(out_mask)
            matched_targets.append(masks_tgt.T)

            batch_start = batch_end

        return matched_outputs, matched_targets, indices

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

if __name__ == '__main__':


    matcher = HungarianMatcher()

    pred = torch.rand([4096, 500])
    gt = torch.randint(0, 2000, [4096])
    offset = torch.tensor([2040, 4096], dtype=torch.int32)

    matched_masks, matched_targets, ids = matcher({'output_mask':pred}, 
                                                  {'instance': gt}, 
                                                   offset)
    
    batch_start = 0
    
    # for b in ids:
    #     print(len(b[0]))

    # print(list(zip(ids[0][0], ids[0][1])))

    # for p in zip(*mapping):
    #     print(p)

    # pred = {
    #     "pred_masks": torch.rand([2, 2048, 20], device='cuda')
    # }

    # target=torch.randint(0, 20, [2,2024])

    # matcher = HungarianMatcher()
    # matcher(pred, target, 'pred_masks')

