from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn as nn

from .builder import build_cost
from .builder import MATCHERS

class BaseMatcher(nn.Module, ABC):

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

    def compute_cost_matrix(self, outputs, targets):
        Cs = None

        for cost in self.cost_terms:
            if Cs is None:
                Cs = cost.compute_cost(outputs, targets)
            else:
                for i, n in enumerate(cost.compute_cost(outputs, targets)):
                    Cs[i] += n

        return Cs

    @abstractmethod
    def forward(self, outputs, targets):
        pass

@MATCHERS.register_module()
class HungarianMatcher(BaseMatcher):
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
        super().__init__(
            cost_terms=cost_terms,
            instance_ignore_index=instance_ignore_index
        )

    def forward(self, outputs, targets):
        batch_size = len(outputs['pred_logits'])

        with torch.no_grad():
            C = self.compute_cost_matrix(outputs, targets)
            C = [c.cpu() for c in C]

            indices = []
            for c in C:
                c = c.numpy()
                indices.append(linear_sum_assignment(c))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]