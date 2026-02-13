"""
Hook Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import torch
import numpy as np
from torch import nn
import torchmetrics
from pointcept.utils.registry import Registry


METRICS = Registry("metrics")

class BaseMetric(nn.Module):

    def update(self, pred, target):
        return self.metric.update(pred, target)

    def compute(self):
        return self.metric.compute()
    
    def reset(self):
        return self.metric.reset()

@METRICS.register_module
class AveragePrecision(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.metric = torchmetrics.AveragePrecision(**kwargs)

    def update(self, pred, target):
        target = target.long()
        return self.metric.update(pred, target)

@METRICS.register_module
class BinaryAccuracy(BaseMetric):
    def __init__(self, average='macro', ignore_index=None, use_logits=False):
       super().__init__()
       self.use_logits = use_logits
       self.metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes=2, 
            average=average, 
            ignore_index=ignore_index
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if self.use_logits:
            preds_2_channels = torch.cat([-preds, preds], dim=1)
        else:
            preds_2_channels = torch.cat([1 - preds, preds], dim=1)
        self.metric.update(preds_2_channels, target)

class BestPRBase(BaseMetric):
    _cache = None
    _last_update_count = -1

    def __init__(self):
        super().__init__()
        self.metric = torchmetrics.classification.BinaryPrecisionRecallCurve()
        self.update_count = 0

    def update(self, pred, target):
        self.update_count += 1
        self.metric.update(pred.detach(), target.long().detach())

    def _get_best_results(self):
        if BestPRBase._last_update_count == self.update_count and BestPRBase._cache is not None:
            return BestPRBase._cache

        p, r, t = self.metric.compute()
        f1 = (2 * p * r) / (p + r + 1e-8)
        idx = torch.argmax(f1)
        
        BestPRBase._cache = {
            'precision': p[idx],
            'recall': r[idx],
            'f1': f1[idx],
            'threshold': t[idx] if idx < len(t) else t[-1]
        }
        BestPRBase._last_update_count = self.update_count
        return BestPRBase._cache

    def reset(self):
        super().reset()
        self.metric.reset()
        self.update_count = 0
        BestPRBase._cache = None
        BestPRBase._last_update_count = -1

@METRICS.register_module
class PrecisionAtBest(BestPRBase):
    def compute(self):
        return self._get_best_results()['precision']

@METRICS.register_module
class RecallAtBest(BestPRBase):
    def compute(self):
        return self._get_best_results()['recall']

@METRICS.register_module
class F1ScoreAtBest(BestPRBase):
    def compute(self):
        return self._get_best_results()['f1']

@METRICS.register_module
class ThresholdAtBest(BestPRBase):
    def compute(self):
        return self._get_best_results()['threshold']

@METRICS.register_module  
class BestF1Bundle(BaseMetric):
    def __init__(self, use_logits=True):
        super().__init__()
        self.use_logits = use_logits
        self.pr_curve = torchmetrics.classification.BinaryPrecisionRecallCurve()

    def update(self, pred, target):
        pred = pred.detach()
        target = target.long().detach()
        
        self.pr_curve.update(pred, target)

    def compute(self):        
        precision, recall, thresholds = self.pr_curve.compute()
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        best_idx = torch.argmax(f1_scores)

        ap = -torch.sum((recall[1:] - recall[:-1]) * precision[1:])
        
        results = {
            "AP": ap,
            "Best_F1": f1_scores[best_idx],
            "Best_Precision": precision[best_idx],
            "Best_Recall": recall[best_idx],
            "Best_Threshold": thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1].item()
        }
        return results

    def reset(self):
        self.acc_metric.reset()
        self.per_class_acc.reset()
        self.pr_curve.reset()
        self.ap_metric.reset()

@METRICS.register_module
class BinaryStatScores(BaseMetric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.metric = torchmetrics.classification.BinaryStatScores(threshold=threshold)

    def compute(self):
        stats = self.metric.compute()

        return dict (
            tp = stats[0],
            fp = stats[1],
            tn = stats[2],
            fn = stats[3]
        )

@METRICS.register_module

# @METRICS.register_module
class InstanceAveragePrecision(BaseMetric):
    def __init__(self, num_classes, class_names, segment_ignore_index=[-1], instance_ignore_index=-1, min_region_size=100, overlaps=None, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.overlaps = overlaps if overlaps is not None else np.sort(np.concatenate(([0.25], np.arange(0.5, 0.951, 0.05))))
        self.device = device
        self.valid_class_names = [name for name in class_names if name not in segment_ignore_index] 
        self.min_region_size = min_region_size
        
        self.metrics = torch.nn.ModuleDict()
        for label in class_names:
            self.metrics[label] = torch.nn.ModuleDict({
                f"ap_{int(ov*100)}": torchmetrics.classification.BinaryAveragePrecision() for ov in self.overlaps
            })
        self.metrics.to(device)

    def _assign_per_class_instances(self, pred, gt):
        pred_instances = {cls: [] for cls in self.valid_class_names}
        gt_instances = {cls: [] for cls in self.valid_class_names}

        instance_ids, idx, counts = np.unique(
            gt['instance'], return_index=True, return_counts=True
        )
        semantic_ids = gt['segment'][idx]

        void_mask = np.in1d(gt['segment'], self.segment_ignore_index)

        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if semantic_ids[i] in self.segment_ignore_index:
                continue
            if counts[i] < self.min_region_size:        
                continue

            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = semantic_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["mask"] = gt['instance'] == instance_ids[i]
            gt_instances[self.class_names[semantic_ids[i]]].append(gt_inst)

        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue

            pred_inst = dict()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(np.logical_and(void_mask, pred_inst["mask"]))

            if pred_inst["vert_count"] < self.min_region_size:   # ← filtriraj male predikcije
                continue

            instance_id += 1
            pred_instances[self.class_names[pred["pred_classes"][i]]].append(pred_inst)

        return pred_instances, gt_instances

    def _compute_iou_matrix(self, preds, gts):
        iou_matrix = np.zeros((len(preds), len(gts)), dtype=np.float32)

        for gt_idx, gt in enumerate(gts):
            for pred_idx, pred in enumerate(preds):
                intersections = np.logical_and(pred['mask'], gt['mask']).sum()
                unions = np.logical_or(pred['mask'], gt['mask']).sum()
                iou = intersections / unions if unions > 0 else 0
                iou_matrix[pred_idx, gt_idx] = iou

        return iou_matrix

    def _match_instances(self, pred_instances, gt_instances, ious, ov_th=0.5):
        matches = {}
        for cls in self.valid_class_names:
            preds = pred_instances[cls]
            gts = gt_instances[cls]
            y_true = []
            y_score = []
            matched_gt = set()

            pred_indices = np.argsort([p['confidence'] for p in preds])[::-1]

            for p_idx in pred_indices:
                best_iou = -1
                best_index = -1

                for g_idx, gt in enumerate(gts):
                    if gt['instance_id'] in matched_gt:
                        continue

                    iou = ious[cls][p_idx, g_idx]
                    if iou > best_iou and iou >= ov_th:
                        best_iou = iou
                        best_index = g_idx

                if best_index != -1:
                    y_true.append(1)
                    matched_gt.add(gts[best_index]['instance_id'])
                else:
                    y_true.append(0)
                    
                y_score.append(preds[p_idx]['confidence'])

            num_fn = len(gts) - len(matched_gt)
            for _ in range(num_fn):
                y_true.append(1)
                y_score.append(np.float32("-inf"))

            matches[cls] = {
                "y_true": np.array(y_true),
                "y_score": np.array(y_score)
            }

        return matches

    def update(self, pred, gt):

        for key in 'pred_classes', 'pred_scores', 'pred_masks':
            if key not in pred:
                raise ValueError(f"Missing key '{key}' in predictions")
            
        for key in 'segment', 'instance':
            if key not in gt:
                raise ValueError(f"Missing key '{key}' in ground truth")

        pred_instances, gt_instances = self._assign_per_class_instances(pred, gt)
        ious = {cls: self._compute_iou_matrix(pred_instances[cls], gt_instances[cls]) for cls in self.valid_class_names}

        for cls in self.class_names:
            if cls not in pred_instances:
                pred_instances[cls] = []
            if cls not in gt_instances:
                gt_instances[cls] = []


        for ov_th in self.overlaps:
            matches_per_ov = self._match_instances(pred_instances, gt_instances, ious, ov_th=ov_th)
            for cls in self.class_names:
                matches = matches_per_ov[cls]
                y_true = matches["y_true"]
                y_score = matches["y_score"]

                if len(y_true) > 0:
                    self.metrics[cls][f"ap_{int(ov_th*100)}"].update(
                        torch.tensor(y_score, device=self.device),
                        torch.tensor(y_true, device=self.device)
                    )

    def compute(self):
        results = {}
        
        for cls, metric_dict in self.metrics.items():
            results[cls] = {}
            for ov in self.overlaps:
                key = f"ap_{int(ov*100)}"
                results[cls][key] = metric_dict[key].compute().item()

            results[cls]["AP25"] = results[cls].get("ap_25", 0.0)
            results[cls]["AP50"] = results[cls].get("ap_50", 0.0)

            ap_values = [
                results[cls][f"ap_{int(ov*100)}"]
                for ov in self.overlaps
                if ov >= 0.5
            ]
            results[cls]["AP"] = np.mean(ap_values) if ap_values else 0.0

        valid_classes = [cls for cls in self.class_names if cls in results and results[cls]]
        if valid_classes:
            metric_keys = ["AP25", "AP50", "AP"] + [f"ap_{int(ov*100)}" for ov in self.overlaps]
            for mkey in metric_keys:
                values = [results[cls].get(mkey, 0.0) for cls in valid_classes]
                results[f"m{mkey}"] = float(np.mean(values))

        return results
    
METRICS.register_module(module=torchmetrics.Accuracy, name="Accuracy")
METRICS.register_module(module=torchmetrics.Precision, name="Precision")
METRICS.register_module(module=torchmetrics.Recall, name="Recall")
METRICS.register_module(module=torchmetrics.F1Score, name="F1Score")
METRICS.register_module(module=torchmetrics.AveragePrecision, name="AP")
METRICS.register_module(module=torchmetrics.classification.BinaryAUROC, name="BinaryAUROC")
METRICS.register_module(module=torchmetrics.classification.JaccardIndex, name="PixelIoU")

def build_metrics(cfg):
    metrics = {}
    for metric_name, metric_cfg in cfg.items():
        metrics[metric_name] = METRICS.build(metric_cfg)
    return metrics

