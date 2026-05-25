# matching/costs.py
import torch
from torch import nn
import torch.nn.functional as F
from .builder import COSTS
from abc import ABC, abstractmethod

class CostTerm(nn.Module, ABC):

    def __init__(self, weight=1.0, enabled=True):
        super().__init__()
        self.weight = weight
        self.enabled = enabled

    @abstractmethod
    def compute_cost(self, outputs, target):
        pass

#TODO: BE CAREFUL ABOUT IGNORED INDICES IN TARGETS
@COSTS.register_module()
class ClassCost(CostTerm):

    '''
        Cost term for the semantic classification error.
        Inputs:
            outputs (dict): Dictionary containing model outputs. Must include 'pred_logits' (list of [N, num_classes] tensors) and 'offset' (int).
            targets (dict): Dictionary containing ground truth data. Must include 'instance_segment' (tensor of shape [N]).
        Output:
            C (list): List of cost tensors, one per batch element, each of shape [N].
    '''

    def __init__(self, weight=1.0, enabled=True, use_logits=False):
        super().__init__(weight, enabled)
        self.use_logits = use_logits

    @torch.no_grad()
    def compute_cost(self, outputs, targets):

        C = []

        bs = 0
        for i, be in enumerate(targets['instance_segment_offset']):
            logits = outputs['pred_logits'][i]  
            labels = targets['instance_segment'][bs:be].long()        # per point semantic labels

            prob = logits.softmax(-1) if not self.use_logits else logits
            if not self.use_logits:
                c = -prob[:, labels]              
            else:
                logp = F.log_softmax(logits, -1)
                c = -logp[:, labels]  
                            
            C.append(c * self.weight)
            bs = be

        return C

@COSTS.register_module()
class MaskBCECost(CostTerm):
    
    def __init__(self, weight=1.0, enabled=True, sample_points=None, instance_ignore_index=-1):
        super().__init__(weight, enabled)
        self.sample_points = sample_points
        self.instance_ignore_index = instance_ignore_index

    def batch_sigmoid_ce_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
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

    @torch.no_grad()
    def compute_cost(self, outputs, target):

        C = []

        bs = 0
        for i, be in enumerate(target['offset']):
            pred_masks = outputs['pred_masks'][bs:be]  
            tgt_flat = target['instance'][bs:be]    

            tgt_flat[tgt_flat == self.instance_ignore_index] = -1
            tgt_masks = F.one_hot(tgt_flat + 1)[:, 1:]     

            filter = tgt_masks != self.instance_ignore_index
            if filter.sum() == 0:
                C.append(torch.zeros((pred_masks.shape[1], 0), device=pred_masks.device))
                bs = be
                continue

            if self.sample_points is not None:
                D = pred_masks.shape[-1]
                idx = torch.randperm(D, device=pred_masks.device)[:self.sample_points]
                pred_masks = pred_masks[:, idx]
                tgt_masks = tgt_masks[:, idx]

            C.append(self.batch_sigmoid_ce_loss(pred_masks.T, tgt_masks.float().T) * self.weight)
            bs = be

        return C

@COSTS.register_module()
class MaskDiceCost(CostTerm):

    def __init__(self, weight=1.0, enabled=True, sample_points=None, instance_ignore_index=-1):
        super().__init__(weight, enabled)
        self.sample_points = sample_points
        self.instance_ignore_index = instance_ignore_index

    def batch_dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
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

    @torch.no_grad()
    def compute_cost(self, outputs, target):

        C = []

        bs = 0
        for i, be in enumerate(target['offset']):
            pred_masks = outputs['pred_masks'][bs:be]  
            tgt_flat = target['instance'][bs:be]    

            tgt_flat[tgt_flat == self.instance_ignore_index] = -1
            tgt_masks = F.one_hot(tgt_flat + 1)[:, 1:]     

            filter = tgt_masks != self.instance_ignore_index
            if filter.sum() == 0:
                C.append(torch.zeros((pred_masks.shape[1], 0), device=pred_masks.device))
                bs = be
                continue

            if self.sample_points is not None:
                D = pred_masks.shape[-1]
                idx = torch.randperm(D, device=pred_masks.device)[:self.sample_points]
                pred_masks = pred_masks[:, idx]
                tgt_masks = tgt_masks[:, idx]

            C.append(self.batch_dice_loss(pred_masks.T, tgt_masks.float().T) * self.weight)
            bs = be

        return C
