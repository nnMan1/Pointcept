import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MaskSelector(nn.Module):
    def __init__(self, instance_ignore_index=-1):
        super().__init__()
        self.instance_ignore_index = instance_ignore_index

    def forward(self, preds, targets, indices):

        matched_outputs = []
        matched_targets = []
        matched_sem_outputs = []
        matched_sem_targets = []
        
        bs, ibs = 0, 0
        for i, (be, ibe) in enumerate(zip(targets['offset'], targets['instance_segment_offset'])):

            pred_ids, tgt_ids = indices[i]
            out_mask = preds['pred_masks'][bs:be]
            out_seg = preds['pred_logits'][i]
            tgt_mask = targets['instance'][bs:be]
            tgt_segm = targets['segment'][bs:be]

            instances_seg = targets['instance_segment'][ibs:ibe]

            filter = tgt_mask != self.instance_ignore_index

            if filter.sum() == 0:
                bs, ibs = be, ibe
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

            bs, ibs = be, ibe
                            
        return matched_outputs, matched_targets, matched_sem_outputs, matched_sem_targets, indices
