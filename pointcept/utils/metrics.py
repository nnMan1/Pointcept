"""
Hook Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import torch
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

