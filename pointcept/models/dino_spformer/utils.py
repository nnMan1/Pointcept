import torch
from torch import nn
from torch.nn import functional as F
from pointcept.utils.misc import batch_iou
from sklearn.cluster import DBSCAN
from scipy import sparse
from scipy.spatial import cKDTree
import numpy as np
import torch_scatter


class _EpsGraphClusterer:
    """Connected components of the eps-radius graph, precomputed once per scene.

    Equivalent to DBSCAN(eps, min_samples=1) on any subset of the points: with
    min_samples=1 every point is a core point, so clusters are exactly the
    connected components of the eps-graph induced on the subset. The KD-tree
    and edge list are built once per scene instead of per mask.

    When per-point superpoint ids are given, superpoints are refined into
    eps-connected fragments and the graph is contracted onto them (~1k nodes
    instead of ~50k points with ~50 edges each). Masks produced by superpoint
    voting/expansion are unions of superpoints, so the contraction is exact;
    a per-mask alignment check falls back to the point-level graph otherwise.
    """

    def __init__(self, coord_np, eps, superpoints_np=None):
        self.n = coord_np.shape[0]
        self.pairs = cKDTree(coord_np).query_pairs(r=eps, output_type="ndarray")
        self.frag = None
        if superpoints_np is not None and superpoints_np.shape[0] == self.n:
            sp = superpoints_np
            same_sp = sp[self.pairs[:, 0]] == sp[self.pairs[:, 1]]
            e_in = self.pairs[same_sp]
            adj = sparse.coo_matrix(
                (np.ones(e_in.shape[0], dtype=np.int8), (e_in[:, 0], e_in[:, 1])),
                shape=(self.n, self.n),
            )
            # only intra-superpoint edges -> components are the eps-connected
            # fragments of each superpoint (isolated points = own fragment)
            _, self.frag = sparse.csgraph.connected_components(adj, directed=False)
            self.n_frag = int(self.frag.max()) + 1
            e_out = self.pairs[~same_sp]
            fe = np.stack([self.frag[e_out[:, 0]], self.frag[e_out[:, 1]]], 1)
            self.frag_edges = np.unique(np.sort(fe, axis=1), axis=0)
            # a representative point per fragment, to read mask membership
            self.frag_first = np.zeros(self.n_frag, dtype=np.int64)
            self.frag_first[self.frag[::-1]] = np.arange(self.n - 1, -1, -1)

    def labels(self, active_np):
        """Cluster labels (0..k-1, no noise) for points where active_np is True."""
        if self.frag is not None:
            active_frag = active_np[self.frag_first]
            # exactness requires the mask to be a union of fragments
            if np.array_equal(active_frag[self.frag], active_np):
                nodes = np.where(active_frag)[0]
                remap = np.zeros(self.n_frag, dtype=np.int64)
                remap[nodes] = np.arange(nodes.shape[0])
                fe = self.frag_edges
                e = fe[active_frag[fe[:, 0]] & active_frag[fe[:, 1]]]
                adj = sparse.coo_matrix(
                    (np.ones(e.shape[0], dtype=np.int8), (remap[e[:, 0]], remap[e[:, 1]])),
                    shape=(nodes.shape[0], nodes.shape[0]),
                )
                _, frag_lab = sparse.csgraph.connected_components(adj, directed=False)
                return frag_lab[remap[self.frag[active_np]]]

        n_act = int(active_np.sum())
        idx_map = np.cumsum(active_np) - 1  # dense reindex of active points
        sel = active_np[self.pairs[:, 0]] & active_np[self.pairs[:, 1]]
        e = self.pairs[sel]
        adj = sparse.coo_matrix(
            (np.ones(e.shape[0], dtype=np.int8), (idx_map[e[:, 0]], idx_map[e[:, 1]])),
            shape=(n_act, n_act),
        )
        _, lab = sparse.csgraph.connected_components(adj, directed=False)
        return lab


def mask_nms(masks, scores, iou_threshold):
    """NMS on binary masks using matrix IoU.
    Args:
        masks: (N, P) binary int/float tensor
        scores: (N,) float tensor
        iou_threshold: float
    Returns:
        keep indices into original ordering
    """
    n = masks.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=masks.device)

    order = torch.argsort(-scores)
    masks_float = masks[order].float()

    # Pairwise IoU via matrix multiply
    inter = masks_float @ masks_float.T  # (N, N)
    areas = masks_float.sum(dim=1)       # (N,)
    union = areas[:, None] + areas[None, :] - inter
    ious = inter / (union + 1e-6)

    keep = torch.ones(n, dtype=torch.bool, device=masks.device)
    for i in range(n):
        if not keep[i]:
            continue
        suppress = ious[i, i + 1:] > iou_threshold
        keep[i + 1:][suppress] = False

    return order[keep]

def dbscan_mask_split(full_masks, scores, labels, coord, mask_sigmoid=None, n_point_thr=100, eps=5, min_samples=1, keep='all', rescore='size', superpoints=None):
    """Split each mask into spatially connected components via DBSCAN.

    Takes the *final* parent scores (class score x mask confidence) so that the
    ranking of the strongest component of each mask stays identical to the
    unsplit pipeline — this preserves AP, while the split geometry improves
    per-mask IoU.

    Args:
        keep: 'all'      keep every component as its own prediction
              'largest'  keep only the biggest component per mask (denoise mode)
        rescore: how components are scored relative to the parent score
              'size'     largest component keeps the parent score, smaller ones
                         are scaled by size relative to the largest component
              'parent'   every component inherits the parent score unchanged
              'cluster'  parent score x mean sigmoid over the component's own
                         points (requires mask_sigmoid; legacy behaviour,
                         hurts AP by promoting confident-but-wrong fragments)

    Returns new (full_masks, scores, labels) on CPU with disconnected clusters
    separated and fragments below n_point_thr dropped.
    """
    coord_np = coord.cpu().numpy() if isinstance(coord, torch.Tensor) else coord
    superpoints_np = superpoints.cpu().numpy() if isinstance(superpoints, torch.Tensor) else superpoints
    # min_samples=1 clustering == connected components of the eps-graph, which
    # can be precomputed once for the scene instead of per mask; sklearn DBSCAN
    # stays as the fallback for min_samples > 1 (where core-point logic matters)
    eps_graph = _EpsGraphClusterer(coord_np, eps, superpoints_np) if min_samples == 1 else None
    # collect (point indices, score, label) per surviving component; dense
    # masks are materialized once at the end — building a dense mask per
    # component OOMs when a small eps shatters masks into many fragments
    entries = []

    for i, (mask, score, label) in enumerate(zip(full_masks, scores, labels)):
        active = mask.bool()
        sig = mask_sigmoid[i] if mask_sigmoid is not None else None

        if active.sum() == 0:
            continue

        active_np = active.cpu().numpy()
        if eps_graph is not None:
            clusters = eps_graph.labels(active_np)
        else:
            clusters = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(coord_np[active_np]).labels_
        active_idx = torch.where(active)[0].cpu()
        valid = clusters != -1
        cluster_ids, cluster_sizes = np.unique(clusters[valid], return_counts=True)

        # drop fragments below n_point_thr BEFORE materializing anything
        big = cluster_sizes > n_point_thr
        cluster_ids, cluster_sizes = cluster_ids[big], cluster_sizes[big]

        if len(cluster_ids) == 0:
            continue

        if keep == 'largest':
            order = [int(np.argmax(cluster_sizes))]
        else:
            order = np.argsort(-cluster_sizes)

        largest_size = float(cluster_sizes.max())
        for ci in order:
            cid = cluster_ids[ci]
            cluster_idx = active_idx[torch.from_numpy(clusters == cid)]

            if rescore == 'cluster':
                assert sig is not None, "rescore='cluster' requires mask_sigmoid"
                new_score = score * sig.cpu()[cluster_idx].mean()
            elif rescore == 'size':
                new_score = score * (float(cluster_sizes[ci]) / largest_size)
            else:  # 'parent'
                new_score = score

            entries.append((cluster_idx, new_score, label))

    if len(entries) == 0:
        empty = full_masks[:0].cpu()
        return empty, scores[:0].cpu(), labels[:0].cpu()

    out_masks = torch.zeros(len(entries), full_masks.shape[1], dtype=full_masks.dtype)
    for j, (cluster_idx, _, _) in enumerate(entries):
        out_masks[j, cluster_idx] = 1

    out_scores = torch.stack([torch.as_tensor(e[1], dtype=torch.float32) for e in entries])
    out_labels = torch.stack([e[2] for e in entries])
    return out_masks, out_scores, out_labels

# ---------------------------------------------------------------------------
# Superpoint voting: how per-point mask predictions are snapped to superpoints
# at inference. Registered by name so new schemes can be A/B-tested from the
# config (mask_selection=dict(voting=dict(type='...', ...))) without touching
# the model. Every function takes per-point predictions and returns per-point
# binary masks that are unions of superpoints.
# ---------------------------------------------------------------------------
SUPERPOINT_VOTERS = {}


def register_voter(name):
    def deco(fn):
        SUPERPOINT_VOTERS[name] = fn
        return fn
    return deco


def _scatter_mean(values, superpoints, n_sp):
    """values: (n_masks, n_points) -> (n_masks, n_superpoints) mean per group."""
    return torch_scatter.scatter(
        values.cpu().T.float(), superpoints.cpu(), dim=0, dim_size=n_sp, reduce='mean'
    ).T


@register_voter('majority')
def _vote_majority(binary, sigmoid, superpoints, n_sp, threshold=0.5, **kw):
    """Historic default: a superpoint joins a mask if >threshold of its points voted."""
    frac = _scatter_mean(binary, superpoints, n_sp).to(binary.device)
    return (frac > threshold).int()[:, superpoints]


@register_voter('soft')
def _vote_soft(binary, sigmoid, superpoints, n_sp, threshold=0.5, **kw):
    """Average the mask PROBABILITIES inside each superpoint instead of the hard
    votes: a superpoint of uniformly borderline points is treated differently
    from one mixing confident yes/no points."""
    assert sigmoid is not None, "voting='soft' needs mask probabilities"
    prob = _scatter_mean(sigmoid, superpoints, n_sp).to(binary.device)
    return (prob > threshold).int()[:, superpoints]


@register_voter('argmax')
def _vote_argmax(binary, sigmoid, superpoints, n_sp, threshold=0.5, **kw):
    """Competitive assignment: every superpoint goes to at most ONE mask (the
    one with the highest mean probability, if it clears threshold), so the
    returned masks never overlap."""
    assert sigmoid is not None, "voting='argmax' needs mask probabilities"
    prob = _scatter_mean(sigmoid, superpoints, n_sp).to(binary.device)  # (n_masks, n_sp)
    best = prob.argmax(dim=0)                                           # (n_sp,)
    keep = prob.max(dim=0).values > threshold
    sp_masks = torch.zeros_like(prob, dtype=torch.int32)
    idx = torch.nonzero(keep).flatten()
    sp_masks[best[idx], idx] = 1
    return sp_masks[:, superpoints]


@register_voter('off')
def _vote_off(binary, sigmoid, superpoints, n_sp, **kw):
    """No snapping: keep the raw per-point masks."""
    return binary.int()


def select_masks(out, superpoints, score_thr=0.0, n_point_thr=100, topk=100, nms_thr=None, coord=None, dbscan_eps=5, dbscan_min_samples=1, split_mode='all', rescore='size', voting='majority'):
        # NOTE: coord is expected in voxel/grid units (1mm voxels): the minimum
        # distance between two points is 1, so dbscan_eps must be > 1 or every
        # point becomes its own cluster and masks shatter into singletons.
        pred_labels = out['pred_logits'][0]
        mask_logits = out['pred_masks'].T
        pred_scores = out['pred_score'][0]

        num_class = pred_labels.shape[1] - 1
        num_query = pred_labels.shape[0]

        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores
        labels = torch.arange(num_class, device=scores.device).unsqueeze(0).repeat(num_query, 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(topk, sorted=False)

        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, num_class, rounding_mode='floor')
        topk_masks = mask_logits[topk_idx]
        topk_masks_sigmoid = topk_masks.sigmoid()
        binary_masks = (topk_masks > 0).float()  # [n_p, M]
        mask_scores = (topk_masks_sigmoid * binary_masks).sum(1) / (binary_masks.sum(1) + 1e-6)
        scores = scores * mask_scores

        # sigmoid values are only needed for rescore='cluster'; skip the
        # (n_p, N) float allocation otherwise
        voting_cfg = dict(type=voting) if isinstance(voting, str) else dict(voting)
        voting_type = voting_cfg.pop('type', 'majority')
        if voting_type not in SUPERPOINT_VOTERS:
            raise KeyError(f"unknown voting '{voting_type}'; have {sorted(SUPERPOINT_VOTERS)}")
        need_sigmoid = rescore == 'cluster' or voting_type in ('soft', 'argmax')

        # expand to full points via superpoints
        if superpoints.max() + 1 == mask_logits.shape[1]:
            # masks are per-superpoint (or identity): expand by indexing
            full_masks = binary_masks[:, superpoints].int()
            full_masks_sigmoid = topk_masks_sigmoid[:, superpoints] if need_sigmoid else None
        else:
            # masks are per-point: snap them to superpoints with the configured
            # voting scheme (see SUPERPOINT_VOTERS above)
            full_masks = SUPERPOINT_VOTERS[voting_type](
                binary_masks, topk_masks_sigmoid, superpoints,
                int(superpoints.max()) + 1, **voting_cfg,
            )
            full_masks_sigmoid = topk_masks_sigmoid if need_sigmoid else None  # already per-point

        # score_thr
        score_mask = scores > score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        full_masks = full_masks[score_mask]  # (n_p, N)
        if full_masks_sigmoid is not None:
            full_masks_sigmoid = full_masks_sigmoid[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = full_masks.sum(1)
        npoint_mask = mask_pointnum > n_point_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        full_masks = full_masks[npoint_mask]  # (n_p, N)
        if full_masks_sigmoid is not None:
            full_masks_sigmoid = full_masks_sigmoid[npoint_mask]  # (n_p, N)

        # everything downstream (DBSCAN split, NMS, numpy conversion) is
        # CPU-bound; move off the GPU and free the large intermediates so the
        # per-component masks built during the split don't OOM on large scenes
        scores = scores.cpu()
        labels = labels.cpu()
        full_masks = full_masks.cpu()
        if full_masks_sigmoid is not None:
            full_masks_sigmoid = full_masks_sigmoid.cpu()
        del topk_masks, topk_masks_sigmoid, binary_masks

        # early return if nothing survives filtering
        if full_masks.shape[0] == 0:
            return dict(
                pred_masks=np.empty((0, full_masks.shape[1]), dtype=np.int32),
                pred_scores=np.empty(0, dtype=np.float32),
                pred_classes=np.empty(0, dtype=np.int64),
            )

        if coord is not None and split_mode != 'off':
            # pass the final scores (class x mask confidence): the largest
            # component of each mask keeps its parent's rank, so the top of the
            # ranking matches the unsplit pipeline (AP) while the split
            # geometry improves per-mask IoU (gt mIoU)
            full_masks, scores, labels = dbscan_mask_split(
                full_masks, scores, labels, coord,
                mask_sigmoid=full_masks_sigmoid,
                n_point_thr=n_point_thr, eps=dbscan_eps, min_samples=dbscan_min_samples,
                keep=split_mode, rescore=rescore, superpoints=superpoints,
            )

        if nms_thr is not None and full_masks.shape[0] > 0:
            nms_idx = mask_nms(full_masks, scores, nms_thr)
            scores = scores[nms_idx]  # (n_p,)
            labels = labels[nms_idx]  # (n_p,)
            full_masks = full_masks[nms_idx]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = full_masks.cpu().numpy()

        return dict(
            pred_masks=mask_pred,
            pred_scores=score_pred,
            pred_classes=cls_pred,
        )

def compute_stats(masks, data, offset, instance_ignore_index=-1):
        
        return_dict = {}
        
        return_dict['pred_scores'] = torch.zeros(len(offset), masks['output_class'].shape[1],  masks['output_class'].shape[2] - 1)
        return_dict['stability_score'] = torch.zeros(len(offset), masks['output_mask'].shape[1])
        return_dict['bious'] =  torch.zeros(len(offset), masks['output_mask'].shape[1], device='cuda')
        batch_start = 0
        
        for i, batch_end in enumerate(offset):
            m = masks['output_mask'][batch_start:batch_end].clone()
            m = F.sigmoid(m)
            t = data['instance'][batch_start:batch_end].clone()

            valid = t != instance_ignore_index

            if valid.sum() > 0:
                m_filtered = m[valid]
                t_filtered = t[valid]
                t_filtered = F.one_hot(t_filtered + 1).float()[:, 1:]
                biou = batch_iou((m_filtered.T > 0.5).float(), t_filtered.T)
                return_dict['bious'][i] = biou.max(-1)[0]
            else:
                return_dict['bious'][i] = torch.zeros(m.shape[1], device='cuda')

            return_dict['stability_score'][i] = calculate_stability_score(m, 0.5, 0.3)
            return_dict['pred_scores'][i] = (masks['output_class'][i].softmax(-1) * masks['output_score'][i])[..., :-1]
            batch_start = batch_end

        return return_dict
   
def calculate_stability_score(masks: torch.Tensor, mask_threshold: float, threshold_offset: float) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(0, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(0, dtype=torch.int32)
    )
    return intersections / (unions + 1e-15)

def db_scan(data, preds):

    new_preds = {
                        "pred_masks": list(),
                        "pred_logits": list(),
                    }

    clsses = preds['output_class'][0]
    masks = preds['output_mask'].T
    coords = data['coord']

    for mask, cls in zip(masks, clsses):

        curr_masks = mask > 0
        
        if coords[curr_masks].shape[0] > 0:
            clusters = (
                                DBSCAN(
                                    eps=0.95,
                                    min_samples=1,
                                    n_jobs=-1,
                                )
                                .fit(coords[curr_masks].cpu())
                                .labels_
                            )
            
            new_mask = torch.zeros(curr_masks.shape, dtype=int, device=curr_masks.device)
            new_mask[curr_masks] = (torch.from_numpy(clusters).to(curr_masks.device) + 1)
            

            for cluster_id in np.unique(clusters):
                if cluster_id != -1:
                    new_preds["pred_masks"].append(
                        mask * (new_mask == cluster_id + 1)
                    )
                    new_preds["pred_logits"].append(
                        cls
                    )

    if len(new_preds['pred_masks']) == 0:
        return preds

    preds['output_class'] = torch.stack(new_preds['pred_logits']).to(preds['output_class'].device)[None, :]
    preds['output_mask'] = torch.stack(new_preds['pred_masks']).T.to(preds['output_class'].device)

    return preds