"""
Hook Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import torch
import torch.distributed as dist
import numpy as np
from torch import nn
import torchmetrics
import pointcept.utils.comm as comm
from pointcept.utils.registry import Registry


METRICS = Registry("metrics")

class BaseMetric(nn.Module):

    def update(self, pred, target):
        return self.metric.update(pred, target)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        return self.metric.reset()

    def sync(self):
        """Aggregate local state across DDP ranks. Default is a no-op."""
        return

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
    def __init__(self):
        super().__init__()
        self.metric = torchmetrics.classification.BinaryPrecisionRecallCurve()
        self.update_count = 0
        self._cache = None
        self._last_update_count = -1

    def update(self, pred, target):
        self.update_count += 1
        self.metric.update(pred.detach(), target.long().detach())

    def _get_best_results(self):
        if self._last_update_count == self.update_count and self._cache is not None:
            return self._cache

        p, r, t = self.metric.compute()
        f1 = (2 * p * r) / (p + r + 1e-8)
        idx = torch.argmax(f1)
        
        self._cache = {
            'precision': p[idx],
            'recall': r[idx],
            'f1': f1[idx],
            'threshold': t[idx] if idx < len(t) else t[-1]
        }
        self._last_update_count = self.update_count
        return self._cache

    def reset(self):
        super().reset()
        self.metric.reset()
        self.update_count = 0
        self._cache = None
        self._last_update_count = -1

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
        self.pr_curve.reset()

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

class InstanceMatcher:
    def __init__(
        self,
        class_names,
        valid_class_names=None,
        segment_ignore_index=(-1,),
        instance_ignore_index=-1,
        min_region_size=100
    ):
        self.class_names = class_names
        self.valid_class_names = valid_class_names or [
            name
            for idx, name in enumerate(class_names)
            if idx not in segment_ignore_index
        ]
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.min_region_size = min_region_size
        
    def _assign_per_class_instances(self, pred, gt):
        pred_instances = {cls: [] for cls in self.valid_class_names}
        gt_instances = {cls: [] for cls in self.valid_class_names}

        instance_ids, idx, counts = np.unique(
            gt['instance'], return_index=True, return_counts=True
        )
        semantic_ids = gt['segment'][idx]

        void_mask = np.in1d(gt['segment'], self.segment_ignore_index)

        # Ground truth instances
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if semantic_ids[i] in self.segment_ignore_index:
                continue
            if counts[i] < self.min_region_size:
                continue

            gt_inst = {
                "instance_id": instance_ids[i],
                "segment_id": semantic_ids[i],
                "dist_conf": 0.0,
                "med_dist": -1.0,
                "vert_count": counts[i],
                "mask": gt['instance'] == instance_ids[i]
            }
            cls_name = self.class_names[semantic_ids[i]]
            gt_instances[cls_name].append(gt_inst)

        # Predictions
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue

            pred_inst = {
                "instance_id": instance_id,
                "segment_id": pred["pred_classes"][i],
                "confidence": pred["pred_scores"][i],
                "mask": np.not_equal(pred["pred_masks"][i], 0),
            }
            raw_vert_count = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            # Discount void overlap so predictions on ignored regions don't
            # get penalized as FPs (ScanNet convention).
            pred_inst["vert_count"] = raw_vert_count - pred_inst["void_intersection"]

            if pred_inst["vert_count"] < self.min_region_size:
                continue

            cls_name = self.class_names[pred["pred_classes"][i]]
            pred_instances[cls_name].append(pred_inst)
            instance_id += 1

        return pred_instances, gt_instances

    def _compute_iou_matrix(self, preds, gts):
        if not preds or not gts:
            return np.zeros((len(preds), len(gts)), dtype=np.float32)

        p_masks = np.stack([p['mask'] for p in preds]).astype(np.float32)
        g_masks = np.stack([g['mask'] for g in gts]).astype(np.float32)

        intersection = np.dot(p_masks, g_masks.T)

        # Discount each prediction's void overlap from its mask size so void
        # points neither inflate the union nor the FP count (ScanNet convention).
        p_void = np.array(
            [p.get("void_intersection", 0) for p in preds], dtype=np.float32
        )
        p_sums = (p_masks.sum(axis=1) - p_void)[:, None]
        g_sums = g_masks.sum(axis=1)[None, :]

        union = p_sums + g_sums - intersection
        return intersection / (union + 1e-6)

    def match_at_threshold(self, pred_instances, gt_instances, ious_per_class, ov_th):
        matches = {}
        iou_sums = {}
        tp_counts = {}

        for cls in self.valid_class_names:
            preds = pred_instances.get(cls, [])
            gts = gt_instances.get(cls, [])

            if not preds and not gts:
                matches[cls] = {"y_true": np.array([]), "y_score": np.array([]), "num_gt": 0}
                continue

            y_true = []
            y_score = []
            # Point count of the matched GT (TP) or -1 (FP), and of the
            # prediction itself — used by scale-stratified AP to bin matches.
            gt_size = []
            pred_size = []
            matched_gt = set()

            # Sort descending by confidence
            pred_indices = np.argsort([p['confidence'] for p in preds])[::-1]

            for p_idx in pred_indices:
                best_iou = -1
                best_index = -1

                for g_idx, gt in enumerate(gts):
                    if gt['instance_id'] in matched_gt:
                        continue
                    iou = ious_per_class[cls][p_idx, g_idx]
                    if iou > best_iou and iou >= ov_th:
                        best_iou = iou
                        best_index = g_idx

                if best_index != -1:
                    y_true.append(1)
                    matched_gt.add(gts[best_index]['instance_id'])
                    iou_sums[cls] = iou_sums.get(cls, 0.0) + best_iou
                    tp_counts[cls] = tp_counts.get(cls, 0) + 1
                    gt_size.append(gts[best_index]['vert_count'])
                else:
                    y_true.append(0)
                    gt_size.append(-1)

                y_score.append(preds[p_idx]['confidence'])
                pred_size.append(preds[p_idx]['vert_count'])

            matches[cls] = {
                "y_true": np.array(y_true),
                "y_score": np.array(y_score),
                "gt_size": np.array(gt_size),
                "pred_size": np.array(pred_size),
                "num_gt": len(gts),
            }

        return matches, iou_sums, tp_counts

    def assign(self, pred, gt):
        pred_instances, gt_instances = self._assign_per_class_instances(pred, gt)
        
        ious = {}
        for cls in self.valid_class_names:
            preds = pred_instances.get(cls, [])
            gts = gt_instances.get(cls, [])
            ious[cls] = self._compute_iou_matrix(preds, gts)

        return {
            "pred_instances": pred_instances,
            "gt_instances": gt_instances,
            "ious": ious,
        }

    def get_matches_at_threshold(self, assignments, ov_th):
        return self.match_at_threshold(
            assignments["pred_instances"],
            assignments["gt_instances"],
            assignments["ious"],
            ov_th=ov_th
        )

# @METRICS.register_module
class InstanceAveragePrecision(BaseMetric):
    
    def __init__(self, num_classes, class_names, segment_ignore_index=(-1,), instance_ignore_index=-1, min_region_size=100, overlaps=None, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.overlaps = overlaps if overlaps is not None else np.sort(np.concatenate(([0.25], np.arange(0.5, 0.951, 0.05))))
        self.device = device
        self.valid_class_names = [
            name
            for idx, name in enumerate(class_names)
            if idx not in segment_ignore_index
        ]

        self.matcher = InstanceMatcher(
            class_names=class_names,
            valid_class_names=self.valid_class_names,
            segment_ignore_index=segment_ignore_index,
            instance_ignore_index=instance_ignore_index,
            min_region_size=min_region_size
        )

        # ScanNet-style AP: akumuliramo y_true/y_score po klasi/pragu, num_gt po klasi.
        self._reset_buffers()

    def _reset_buffers(self):
        self.y_true = {cls: {ov: [] for ov in self.overlaps} for cls in self.class_names}
        self.y_score = {cls: {ov: [] for ov in self.overlaps} for cls in self.class_names}
        # num_gt po klasi je nezavisan od praga
        self.num_gt = {cls: 0 for cls in self.class_names}

    def update(self, pred_dict, gt_dict):
        for key in 'pred_classes', 'pred_scores', 'pred_masks':
            if key not in pred_dict:
                raise ValueError(f"Missing key '{key}' in predictions")

        for key in 'segment', 'instance':
            if key not in gt_dict:
                raise ValueError(f"Missing key '{key}' in ground truth")

        assignments = self.matcher.assign(pred_dict, gt_dict)
        gt_instances = assignments["gt_instances"]

        for cls in self.class_names:
            self.num_gt[cls] += len(gt_instances.get(cls, []))

        for ov_th in self.overlaps:
            matches_per_ov, _, _ = self.matcher.get_matches_at_threshold(assignments, ov_th=ov_th)
            for cls in self.class_names:
                matches = matches_per_ov.get(cls, {"y_true": np.array([]), "y_score": np.array([])})
                y_true = matches["y_true"]
                y_score = matches["y_score"]
                if len(y_true) > 0:
                    self.y_true[cls][ov_th].extend(y_true.tolist())
                    self.y_score[cls][ov_th].extend(y_score.tolist())

    @staticmethod
    def _voc_ap(y_true, y_score, num_gt):
        """VOC-style all-point interpolation AP (ScanNet konvencija).
        Recall se normalizuje sa num_gt, ne sa len(y_true)."""
        if num_gt == 0:
            return 0.0
        if len(y_true) == 0:
            return 0.0

        y_true = np.asarray(y_true, dtype=np.float64)
        y_score = np.asarray(y_score, dtype=np.float64)

        order = np.argsort(-y_score, kind="stable")
        y_true_sorted = y_true[order]

        cum_tp = np.cumsum(y_true_sorted == 1)
        cum_fp = np.cumsum(y_true_sorted == 0)

        recall = cum_tp / float(num_gt)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # Monotono opadajuća precision od desna ka levo
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        change = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))
        return ap

    def sync(self):
        if comm.get_world_size() == 1:
            return
        gathered = comm.all_gather(
            {"y_true": self.y_true, "y_score": self.y_score, "num_gt": self.num_gt}
        )
        merged_y_true = {cls: {ov: [] for ov in self.overlaps} for cls in self.class_names}
        merged_y_score = {cls: {ov: [] for ov in self.overlaps} for cls in self.class_names}
        merged_num_gt = {cls: 0 for cls in self.class_names}
        for state in gathered:
            for cls in self.class_names:
                merged_num_gt[cls] += state["num_gt"][cls]
                for ov in self.overlaps:
                    merged_y_true[cls][ov].extend(state["y_true"][cls][ov])
                    merged_y_score[cls][ov].extend(state["y_score"][cls][ov])
        self.y_true = merged_y_true
        self.y_score = merged_y_score
        self.num_gt = merged_num_gt

    def compute(self):
        results = {}

        for cls in self.class_names:
            results[cls] = {}
            for ov in self.overlaps:
                ap = self._voc_ap(self.y_true[cls][ov], self.y_score[cls][ov], self.num_gt[cls])
                results[cls][f"ap_{int(ov * 100)}"] = ap

            results[cls]["AP25"] = results[cls].get("ap_25", 0.0)
            results[cls]["AP50"] = results[cls].get("ap_50", 0.0)

            ap_values = [
                results[cls][f"ap_{int(ov * 100)}"]
                for ov in self.overlaps
                if ov >= 0.5
            ]
            results[cls]["AP"] = float(np.mean(ap_values)) if ap_values else 0.0

        valid_classes = [cls for cls in self.valid_class_names if cls in results and self.num_gt.get(cls, 0) > 0]
        if valid_classes:
            metric_keys = ["AP25", "AP50", "AP"] + [f"ap_{int(ov * 100)}" for ov in self.overlaps]
            for mkey in metric_keys:
                values = [results[cls].get(mkey, 0.0) for cls in valid_classes]
                results[f"m{mkey}"] = float(np.mean(values))

        return results

    def reset(self):
        self._reset_buffers()

# @METRICS.register_module
class ScaleStratifiedInstanceAveragePrecision(BaseMetric):
    """Scale-stratified instance AP: AP_S, AP_M and AP_L.

    GT instances are partitioned by point count into Small / Medium / Large:
        |X_p| <= tau_S            -> S
        tau_S < |X_p| <= tau_L    -> M
        |X_p| > tau_L             -> L
    By default tau_S and tau_L are the 33rd and 66th percentiles of GT
    instance sizes pooled across the whole evaluation set, computed lazily in
    ``compute()`` (so the thresholds reflect the full test set rather than any
    single scene). They can be pinned explicitly via ``size_thresholds``.

    Matching is identical to :class:`InstanceAveragePrecision` (ScanNet-style,
    greedy, one prediction per GT). Following COCO's area-range convention,
    each size bin is evaluated independently: true positives are assigned to
    the bin of their *matched GT*, unmatched predictions (false positives) are
    assigned to the bin of their *own* point count, and matches falling outside
    the bin are ignored. AP per bin is the VOC all-point AP averaged over
    overlaps >= 0.5 and over the valid classes (mean AP).
    """

    SIZE_BINS = ("S", "M", "L")

    def __init__(self, num_classes, class_names, segment_ignore_index=(-1,),
                 instance_ignore_index=-1, min_region_size=100, overlaps=None,
                 size_percentiles=(33, 66), size_thresholds=None,
                 device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.overlaps = overlaps if overlaps is not None else np.sort(
            np.concatenate(([0.25], np.arange(0.5, 0.951, 0.05)))
        )
        self.size_percentiles = size_percentiles
        # Optional explicit (tau_S, tau_L); overrides the percentile estimate.
        self.size_thresholds = size_thresholds
        self.device = device
        self.valid_class_names = [
            name
            for idx, name in enumerate(class_names)
            if idx not in segment_ignore_index
        ]

        self.matcher = InstanceMatcher(
            class_names=class_names,
            valid_class_names=self.valid_class_names,
            segment_ignore_index=segment_ignore_index,
            instance_ignore_index=instance_ignore_index,
            min_region_size=min_region_size
        )

        self._reset_buffers()

    def _reset_buffers(self):
        # Per class / per overlap parallel arrays of matched predictions.
        self.records = {
            cls: {
                ov: {"y_true": [], "y_score": [], "gt_size": [], "pred_size": []}
                for ov in self.overlaps
            }
            for cls in self.class_names
        }
        # Point counts of every GT instance, per class (for binning + num_gt).
        self.gt_sizes = {cls: [] for cls in self.class_names}

    def update(self, pred_dict, gt_dict):
        for key in 'pred_classes', 'pred_scores', 'pred_masks':
            if key not in pred_dict:
                raise ValueError(f"Missing key '{key}' in predictions")
        for key in 'segment', 'instance':
            if key not in gt_dict:
                raise ValueError(f"Missing key '{key}' in ground truth")

        assignments = self.matcher.assign(pred_dict, gt_dict)
        gt_instances = assignments["gt_instances"]

        for cls in self.class_names:
            for gt in gt_instances.get(cls, []):
                self.gt_sizes[cls].append(int(gt["vert_count"]))

        for ov_th in self.overlaps:
            matches_per_ov, _, _ = self.matcher.get_matches_at_threshold(assignments, ov_th=ov_th)
            for cls in self.class_names:
                matches = matches_per_ov.get(cls)
                if not matches or len(matches["y_true"]) == 0:
                    continue
                rec = self.records[cls][ov_th]
                rec["y_true"].extend(matches["y_true"].tolist())
                rec["y_score"].extend(matches["y_score"].tolist())
                rec["gt_size"].extend(matches["gt_size"].tolist())
                rec["pred_size"].extend(matches["pred_size"].tolist())

    @staticmethod
    def _bin_mask(sizes, bin_name, tau_s, tau_l):
        if bin_name == "S":
            return sizes <= tau_s
        if bin_name == "M":
            return (sizes > tau_s) & (sizes <= tau_l)
        return sizes > tau_l

    def _select_for_bin(self, rec, bin_name, tau_s, tau_l):
        y_true = np.asarray(rec["y_true"])
        if y_true.size == 0:
            return np.array([]), np.array([])
        y_score = np.asarray(rec["y_score"])
        gt_size = np.asarray(rec["gt_size"])
        pred_size = np.asarray(rec["pred_size"])
        is_tp = y_true == 1
        # TPs are binned by the GT they matched; FPs by their own size.
        sizes = np.where(is_tp, gt_size, pred_size)
        keep = self._bin_mask(sizes, bin_name, tau_s, tau_l)
        return y_true[keep], y_score[keep]

    def _resolve_thresholds(self):
        if self.size_thresholds is not None:
            return float(self.size_thresholds[0]), float(self.size_thresholds[1])
        pooled = [np.asarray(v, dtype=np.float64) for v in self.gt_sizes.values() if len(v)]
        if not pooled:
            return 0.0, 0.0
        all_sizes = np.concatenate(pooled)
        tau_s, tau_l = np.percentile(all_sizes, self.size_percentiles)
        return float(tau_s), float(tau_l)

    def sync(self):
        if comm.get_world_size() == 1:
            return
        gathered = comm.all_gather({"records": self.records, "gt_sizes": self.gt_sizes})
        self._reset_buffers()
        for state in gathered:
            for cls in self.class_names:
                self.gt_sizes[cls].extend(state["gt_sizes"][cls])
                for ov in self.overlaps:
                    src = state["records"][cls][ov]
                    dst = self.records[cls][ov]
                    for k in ("y_true", "y_score", "gt_size", "pred_size"):
                        dst[k].extend(src[k])

    def compute(self):
        tau_s, tau_l = self._resolve_thresholds()
        results = {"tau_S": tau_s, "tau_L": tau_l}

        for bin_name in self.SIZE_BINS:
            # num_gt per class within this size bin.
            num_gt_bin = {}
            for cls in self.class_names:
                sizes = np.asarray(self.gt_sizes[cls], dtype=np.float64)
                if sizes.size == 0:
                    num_gt_bin[cls] = 0
                else:
                    num_gt_bin[cls] = int(np.count_nonzero(
                        self._bin_mask(sizes, bin_name, tau_s, tau_l)
                    ))

            per_class_ap = {}
            for cls in self.class_names:
                ap_over_ov = []
                for ov in self.overlaps:
                    if ov < 0.5:
                        continue
                    y_true, y_score = self._select_for_bin(
                        self.records[cls][ov], bin_name, tau_s, tau_l
                    )
                    ap_over_ov.append(
                        InstanceAveragePrecision._voc_ap(y_true, y_score, num_gt_bin[cls])
                    )
                per_class_ap[cls] = float(np.mean(ap_over_ov)) if ap_over_ov else 0.0

            results[f"per_class_AP_{bin_name}"] = per_class_ap
            results[f"num_gt_{bin_name}"] = int(sum(
                num_gt_bin[cls] for cls in self.valid_class_names
            ))

            valid = [
                per_class_ap[cls]
                for cls in self.valid_class_names
                if cls in per_class_ap
            ]
            results[f"AP_{bin_name}"] = float(np.mean(valid)) if valid else 0.0

        return results

    def reset(self):
        self._reset_buffers()

# @METRICS.register_module()
class MatchedOnlyInstanceMeanIoU(BaseMetric):

    def __init__(
        self,
        num_classes,
        class_names,
        segment_ignore_index=(-1,),
        instance_ignore_index=-1,
        min_region_size=100,
        overlaps=None,
        device="cuda",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.overlaps = overlaps if overlaps is not None else np.sort(
            np.concatenate(([0.25], np.arange(0.5, 0.951, 0.05)))
        )
        self.device = device
        self.valid_class_names = [
            name
            for idx, name in enumerate(class_names)
            if idx not in segment_ignore_index
        ]

        self.matcher = InstanceMatcher(
            class_names=class_names,
            valid_class_names=self.valid_class_names,
            segment_ignore_index=segment_ignore_index,
            instance_ignore_index=instance_ignore_index,
            min_region_size=min_region_size
        )

        self.iou_sums = {
            label: torch.zeros(len(self.overlaps), dtype=torch.float64, device=device)
            for label in class_names
        }
        self.tp_counts = {
            label: torch.zeros(len(self.overlaps), dtype=torch.long, device=device)
            for label in class_names
        }

    def update(self, pred_dict, gt_dict):
        assignments = self.matcher.assign(pred_dict, gt_dict)

        for i, ov_th in enumerate(self.overlaps):
            matches, iou_sums_ov, tp_counts_ov = self.matcher.get_matches_at_threshold(assignments, ov_th=ov_th)

            for cls in self.class_names:
                self.iou_sums[cls][i] += iou_sums_ov.get(cls, 0.0)
                self.tp_counts[cls][i] += tp_counts_ov.get(cls, 0)

    def sync(self):
        if comm.get_world_size() == 1:
            return
        for cls in self.class_names:
            dist.all_reduce(self.iou_sums[cls], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.tp_counts[cls], op=dist.ReduceOp.SUM)

    def compute(self):
        results = {}

        for cls in self.class_names:
            results[cls] = {}
            for i, ov in enumerate(self.overlaps):
                tp = self.tp_counts[cls][i].item()
                if tp > 0:
                    mean_iou = self.iou_sums[cls][i].item() / tp
                    results[cls][f"MmIoU@{int(ov*100):02d}"] = mean_iou
                else:
                    results[cls][f"MmIoU@{int(ov*100):02d}"] = 0.0

            iou_values = [
                results[cls].get(f"MmIoU@{int(ov*100):02d}", 0.0)
                for ov in self.overlaps
            ]
            results[cls]["MmIoU"] = np.mean(iou_values) if iou_values else 0.0

        valid_classes = [cls for cls in self.class_names if cls in results]
        if valid_classes:
            for i, ov in enumerate(self.overlaps):
                key = f"MmIoU@{int(ov*100):02d}"
                values = [results[cls].get(key, 0.0) for cls in valid_classes]
                results[f"m{key}"] = float(np.mean(values))

            all_values = []
            for cls in valid_classes:
                for ov in self.overlaps:
                    key = f"MmIoU@{int(ov*100):02d}"
                    all_values.append(results[cls].get(key, 0.0))
            results["MmIoU"] = float(np.mean(all_values)) if all_values else 0.0

        return results

    def reset(self):
        for cls in self.class_names:
            self.iou_sums[cls].zero_()
            self.tp_counts[cls].zero_()

# @METRICS.register_module()
class InstanceMeanIoU(BaseMetric):

    def __init__(
        self,
        num_classes,
        class_names,
        segment_ignore_index=(-1,),
        instance_ignore_index=-1,
        min_region_size=100,
        overlaps=None,
        device="cuda",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.overlaps = overlaps if overlaps is not None else np.sort(
            np.concatenate(([0, 0.25], np.arange(0.5, 0.951, 0.05)))
        )
        
        self.device = device
        self.valid_class_names = [
            name
            for idx, name in enumerate(class_names)
            if idx not in segment_ignore_index
        ]

        self.matcher = InstanceMatcher(
            class_names=class_names,
            valid_class_names=self.valid_class_names,
            segment_ignore_index=segment_ignore_index,
            instance_ignore_index=instance_ignore_index,
            min_region_size=min_region_size
        )

        self.iou_sums = {
            label: torch.zeros(len(self.overlaps), dtype=torch.float64, device=device)
            for label in class_names
        }
        self.tp_counts = {
            label: torch.zeros(len(self.overlaps), dtype=torch.long, device=device)
            for label in class_names
        }

    def update(self, pred_dict, gt_dict):
        assignments = self.matcher.assign(pred_dict, gt_dict)
        gt_instances = assignments["gt_instances"]
        for i, ov_th in enumerate(self.overlaps):
            _, iou_sums_ov, _ = self.matcher.get_matches_at_threshold(assignments, ov_th=ov_th)

            for cls in self.class_names:
                self.iou_sums[cls][i] += iou_sums_ov.get(cls, 0.0)
                # Imenitelj = ukupno GT-ova za klasu (ne TP-only) — odrazava i FN-ove kao 0 doprinos
                self.tp_counts[cls][i] += len(gt_instances.get(cls, []))

    def sync(self):
        if comm.get_world_size() == 1:
            return
        for cls in self.class_names:
            dist.all_reduce(self.iou_sums[cls], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.tp_counts[cls], op=dist.ReduceOp.SUM)

    def compute(self):
        results = {}

        for cls in self.class_names:
            results[cls] = {}
            for i, ov in enumerate(self.overlaps):
                tp = self.tp_counts[cls][i].item()
                if tp > 0:
                    mean_iou = self.iou_sums[cls][i].item() / tp
                    results[cls][f"mIoU@{int(ov*100):02d}"] = mean_iou
                else:
                    results[cls][f"mIoU@{int(ov*100):02d}"] = 0.0

            iou_values = [
                results[cls].get(f"mIoU@{int(ov*100):02d}", 0.0)
                for ov in self.overlaps
            ]
            results[cls]["mIoU"] = np.mean(iou_values) if iou_values else 0.0

        valid_classes = [cls for cls in self.class_names if cls in results]
        if valid_classes:
            for i, ov in enumerate(self.overlaps):
                key = f"mIoU@{int(ov*100):02d}"
                values = [results[cls].get(key, 0.0) for cls in valid_classes]
                results[f"m{key}"] = float(np.mean(values))

            all_values = []
            for cls in valid_classes:
                for ov in self.overlaps:
                    key = f"mIoU@{int(ov*100):02d}"
                    all_values.append(results[cls].get(key, 0.0))
            results["mIoU"] = float(np.mean(all_values)) if all_values else 0.0

        return results

    def reset(self):
        for cls in self.class_names:
            self.iou_sums[cls].zero_()
            self.tp_counts[cls].zero_()

class GTInstanceIoU(BaseMetric):

    def __init__(
        self,
        num_classes,
        class_names,
        segment_ignore_index=(-1,),
        instance_ignore_index=-1,
        min_region_size=100,
        device="cuda",
        **kwargs):
    
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.device = device
        self.valid_class_names = [
            name
            for idx, name in enumerate(class_names)
            if idx not in segment_ignore_index
        ]

        self.matcher = InstanceMatcher(
            class_names=class_names,
            valid_class_names=self.valid_class_names,
            segment_ignore_index=segment_ignore_index,
            instance_ignore_index=instance_ignore_index,
            min_region_size=min_region_size
        )

        # Track IoU statistics per class
        self.iou_stats = {label: [] for label in class_names}
        
    def _compute_iou(self, mask1, mask2):
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        return float(intersection) / float(union)
    
    def update(self, pred_dict, gt_dict):
        """
        For each GT instance, find best matching prediction and compute IoU.
        """
        assignments = self.matcher.assign(pred_dict, gt_dict)
        
        pred_instances = assignments["pred_instances"]
        gt_instances = assignments["gt_instances"]
        ious_per_class = assignments["ious"]
        
        for cls in self.class_names:
            preds = pred_instances.get(cls, [])
            gts = gt_instances.get(cls, [])
            iou_matrix = ious_per_class.get(cls, np.array([]))
            
            # For each GT, find best matching prediction
            for gt_idx, gt in enumerate(gts):
                if iou_matrix.size == 0:
                    # No predictions for this class
                    best_iou = 0.0
                else:
                    # Get best IoU for this GT across all predictions
                    best_iou = iou_matrix[:, gt_idx].max()
                
                self.iou_stats[cls].append(best_iou)

    def sync(self):
        if comm.get_world_size() == 1:
            return
        gathered = comm.all_gather(self.iou_stats)
        merged = {cls: [] for cls in self.class_names}
        for state in gathered:
            for cls in self.class_names:
                merged[cls].extend(state.get(cls, []))
        self.iou_stats = merged

    def compute(self):
        """
        Compute mean IoU per class and overall mean IoU.
        """
        results = {}

        for cls in self.class_names:
            results[cls] = {}
            
            if len(self.iou_stats[cls]) > 0:
                mean_iou = float(np.mean(self.iou_stats[cls]))
                results[cls]["mIoU"] = mean_iou
            else:
                results[cls]["mIoU"] = 0.0

        # Compute mean IoU across valid classes
        valid_classes = [cls for cls in self.valid_class_names if cls in results]
        if valid_classes:
            mious = [results[cls]["mIoU"] for cls in valid_classes]
            results["mIoU"] = float(np.mean(mious))
        else:
            results["mIoU"] = 0.0

        return results

    def reset(self):
        """Reset statistics for new evaluation."""
        for cls in self.class_names:
            self.iou_stats[cls] = []


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

