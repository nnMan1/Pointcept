import numpy as np

class ScoreMatcher:

    def __init__(self, num_classes, class_names, segment_ignore_index=-1, instance_ignore_index=-1):
        self.num_classes = num_classes
        self.class_names = class_names
        self.valid_class_names = [cn for cn in class_names if cn not in segment_ignore_index]
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def _compute_iou_matrix(self, preds, gts):
        iou_matrix = np.zeros((len(preds), len(gts)), dtype=np.float32)

        for gt_idx, gt in enumerate(gts):
            for pred_idx, pred in enumerate(preds):
                intersections = np.logical_and(pred['mask'], gt['mask']).sum()
                unions = np.logical_or(pred['mask'], gt['mask']).sum()
                iou = intersections / unions if unions > 0 else 0
                iou_matrix[pred_idx, gt_idx] = iou

        return iou_matrix
    
    def __call__(self, pred, gt):
        for key in 'pred_classes', 'pred_scores', 'pred_masks':
            if key not in pred:
                raise ValueError(f"Missing key '{key}' in predictions")
            
        for key in 'semantic', 'instance':
            if key not in gt:
                raise ValueError(f"Missing key '{key}' in ground truth")
            
        pred_instances, gt_instances = self.per_class_instances(pred, gt)

        matches = {}
        for cls in self.valid_class_names:
            preds = pred_instances[cls]
            gts = gt_instances[cls]
            y_true = []
            y_score = []
            ious = self._compute_iou_matrix(preds, gts)
            matched_gt = set()

            pred_indices = np.argsort([p['confidence'] for p in preds])[::-1]

            for p_idx in pred_indices:
                best_iou = -1
                best_index = -1

                for g_idx, gt in enumerate(gts):
                    if gt['instance_id'] in matched_gt:
                        continue
                    iou = ious[p_idx, g_idx]
                    if iou > best_iou:
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
                y_score.append(0)

            matches[cls] = {
                "y_true": np.array(y_true),
                "y_score": np.array(y_score)
            }
                