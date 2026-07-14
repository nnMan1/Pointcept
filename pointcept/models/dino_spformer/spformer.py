import torch
import torch.nn.functional as F
import torch_scatter
from torch import nn
from torch.amp import autocast

from pointcept.models.builder import MODELS, build_model
from pointcept.positional_embeddings import build_positional_embedding
from pointcept.models.losses import DiceLoss, BinaryFocalLoss
from pointcept.models.utils.nn import (
    GenericMLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer,
    SuperpointPooling, SuperpointUnpooling, pad_data,
)
from .utils import compute_stats, select_masks, db_scan
from pointcept.models.utils.matcher import build_matcher, MaskSelector


class Encoder(nn.Module):

    def __init__(self, backbone, out_channels, backbone_out_channels=64):
        super().__init__()
        self.out_channels = out_channels
        self.backbone = build_model(backbone)
        self.mask_features_head = nn.Sequential(
            nn.Linear(backbone_out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, data_dict):
        offset = data_dict['offset']
        values = self.backbone(data_dict)
        features = self.mask_features_head(values['feat'])
        return {
            'features': features,
            'offset': offset,
            'fts_loss': values.get('loss', 0.0),
        }


class Decoder(nn.Module):

    def __init__(self, in_channels, mask_modules, query_refinement_modules, hlevels, pos_dim=None):
        super().__init__()
        self.mask_features_head = nn.Sequential(
            nn.Linear(in_channels, mask_modules[0]['hidden_dim']),
            nn.ReLU(),
            nn.Linear(mask_modules[0]['hidden_dim'], mask_modules[0]['hidden_dim']),
        )
        self.query_features_head = nn.Sequential(
            nn.Linear(in_channels, query_refinement_modules[0]['in_channels']),
            nn.LayerNorm(query_refinement_modules[0]['in_channels']),
            nn.ReLU(),
        )
        # Project positional encoding into the mask-feature space so the
        # point<->query mask dot-product becomes position-aware (lets the model
        # separate identical-appearance instances). Only built when enabled.
        self.pos_proj = None
        if pos_dim is not None:
            self.pos_proj = nn.Linear(pos_dim, mask_modules[0]['hidden_dim'])
        self.mask_modules = nn.ModuleList([MaskModule(**cfg) for cfg in mask_modules])
        self.query_refinements = nn.ModuleList([QueryRefinement(**c) for c in query_refinement_modules])

    def forward(self, data, query_features):
        offset = data['offset']
        features = data['features']

        mask_point_features = self.mask_features_head(features)
        if self.pos_proj is not None and data.get('positional_embedding') is not None:
            mask_point_features = mask_point_features + self.pos_proj(data['positional_embedding'])
        query_point_features = self.query_features_head(features)

        out = []

        for mask_module in self.mask_modules:
            for _ in range(mask_module.reuse):
                for query_refinement in self.query_refinements:
                    mask_module_data = {
                        'query_feat': query_features,
                        'mask_features': mask_point_features,
                        'offset': offset,
                    }
                    masks = mask_module(mask_module_data)
                    attention_mask = masks['attn_mask']
                    query_features = query_refinement(
                        query_point_features,
                        attention_mask,
                        data['offset'],
                        query_features,
                        pos=data['positional_embedding'],
                    )
                    out.append(masks)

        mask_module_data = {
            'query_feat': query_features,
            'mask_features': mask_point_features,
            'offset': offset,
        }
        masks = mask_module(mask_module_data)
        masks['query_feat'] = query_features  # expose for triplet loss
        masks['query_point_features'] = query_point_features  # expose for prototype building
        out.append(masks)

        return out


class MaskModule(nn.Module):

    def __init__(self, hidden_dim, num_classes, return_attn_masks, reuse=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.reuse = reuse
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.out_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.class_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes),
        )
        self.return_attn_masks = return_attn_masks

    def forward(self, data):
        query_feat, mask_features, offset = data['query_feat'], data['mask_features'], data['offset']

        query_feat = self.decoder_norm(query_feat)
        output_class = self.class_embed_head(query_feat)
        outputs_score = self.out_score(query_feat)

        output_masks = []
        attn_masks = []

        return_dict = {
            'pred_logits': output_class,
            'pred_score': outputs_score,
        }

        bs = 0
        for i, be in enumerate(offset):
            output_masks.append(mask_features[bs:be] @ query_feat[i].T)
            bs = be

        outputs_mask = torch.cat(output_masks)
        return_dict['pred_masks'] = outputs_mask

        if self.return_attn_masks:
            bs = 0
            for be in offset:
                attn_masks.append((outputs_mask[bs:be].sigmoid() < 0.5).bool())
                attn_masks[-1].permute(1, 0)[
                    torch.where(attn_masks[-1].sum(0) == attn_masks[-1].shape[0])
                ] = False
                bs = be
            return_dict['attn_mask'] = torch.cat(attn_masks).detach()

        return return_dict


class QueryRefinement(nn.Module):

    def __init__(self, in_channels, dim_feedforward, mask_dim, pre_norm, num_heads, dropout, sample_size=None):
        super().__init__()
        self.dim_feed_forward = dim_feedforward
        self.mask_dim = mask_dim
        self.pre_norm = pre_norm
        self.num_heads = num_heads
        self.dropout = dropout
        self.sample_size = sample_size

        self.cross_attention = CrossAttentionLayer(
            d_model=self.mask_dim,
            nhead=self.num_heads,
            dropout=self.dropout,
            normalize_before=self.pre_norm,
        )
        self.self_attention = SelfAttentionLayer(
            d_model=self.mask_dim,
            nhead=self.num_heads,
            dropout=self.dropout,
            normalize_before=self.pre_norm,
        )
        self.ffn_attention = FFNLayer(
            d_model=self.mask_dim,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            normalize_before=self.pre_norm,
            activation='gelu',
        )

    def forward(self, point_features, attn_mask, offset, queries, pos):
        point_features, rand_idx, mask_idx = pad_data(point_features, offset, self.sample_size)
        attn_mask, _, _ = pad_data(attn_mask, offset, self.sample_size, rand_idx, mask_idx)

        if pos is not None:
            pos, _, _ = pad_data(pos, offset, self.sample_size, rand_idx, mask_idx)

        m = torch.stack(mask_idx)
        attn_mask = torch.logical_or(attn_mask, m[..., None])
        attn_mask = attn_mask.permute((0, 2, 1))

        output = self.cross_attention(
            query=queries,
            key=point_features,
            value=point_features,
            attn_mask=attn_mask.repeat_interleave(self.num_heads, dim=0),
            pos=pos,
        )
        output = self.self_attention(
            output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
        )
        queries = self.ffn_attention(output)

        return queries


@MODELS.register_module("MySPFormer")
class MySPFormer(nn.Module):

    def __init__(
        self,
        num_query,
        encoder,
        decoder,
        matcher,
        use_superpoint_pooling=True,
        positional_embedding=None,
        use_positional_encoding=None,
        seg_ce_loss_weight=0.5,
        mask_ce_loss_weight=1.0,
        mask_dice_loss_weight=1.0,
        score_loss_weight=0.5,
        fts_loss_weight=0.00,
        triplet_loss_weight=0.0,
        overlap_loss_weight=0.0,
        triplet_margin=0.0,
        border_loss_weight=0.0,
        border_focal_alpha=0.5,
<<<<<<< HEAD
        mask_selection=None,
        eval_superpoint_voting=True,
=======
        equal_instance_weight=False,
>>>>>>> 7fe784f6cd6da9304521f014797b9f4a7dd0c8fc
    ):
        super().__init__()

        self.num_query = num_query
        self.encoder = Encoder(**encoder)

        # eval-time mask filtering (see utils.select_masks); override via config
        self.mask_selection = dict(
            score_thr=0.0,
            n_point_thr=100,
            topk=100,
            nms_thr=None,
            dbscan_eps=5,  # coord is in voxel units (1mm grid): min point distance is 1, so eps must be > 1
            dbscan_min_samples=1,
            split_mode='all',   # 'off' | 'largest' | 'all'
            rescore='size',     # 'size' | 'parent' | 'cluster'
        )
        if mask_selection is not None:
            self.mask_selection.update(mask_selection)

        # Per-point border detection branch (binary: border vs non-border).
        # Built only when enabled so checkpoints without it still load.
        self.border_head = None
        if border_loss_weight > 0:
            border_in = encoder['out_channels']
            self.border_head = nn.Sequential(
                nn.Linear(border_in, border_in),
                nn.ReLU(),
                nn.Linear(border_in, 1),
            )
            self.border_loss = BinaryFocalLoss(alpha=border_focal_alpha)

        self.superpoint_pooling = SuperpointPooling()
        self.superpoint_unpooling = SuperpointUnpooling()

        # Whether to use positional encoding. Defaults to "auto": on iff a
        # positional_embedding config is provided (preserves old behaviour).
        # Set explicitly in the config to force on/off.
        if use_positional_encoding is None:
            use_positional_encoding = positional_embedding is not None
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            assert positional_embedding is not None, \
                "use_positional_encoding=True requires a positional_embedding config"
            # Enable the position-aware mask head in the decoder.
            decoder['pos_dim'] = positional_embedding['d_pos']

        query_dim = decoder['query_refinement_modules'][0]['mask_dim']
        self.__query = nn.Embedding(num_query, decoder['query_refinement_modules'][0]['mask_dim'])
        
        # MLP to project FPS-sampled features into query space
        # self.query_proj = nn.Sequential(
        #     nn.Linear(encoder['out_channels'], query_dim),
        #     nn.LayerNorm(query_dim),
        #     nn.ReLU(),
        #     nn.Linear(query_dim, query_dim),
        # )

        for i, _ in enumerate(decoder['mask_modules']):
            decoder['mask_modules'][i]['num_classes'] += 1

        self.decoder = Decoder(**decoder)
        self.matcher = build_matcher(matcher)
        self.mask_selector = MaskSelector()

        weight = torch.ones(decoder['mask_modules'][0]['num_classes'])
        weight[-1] = 0.1

        self.semantic_ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.mask_dice_loss = DiceLoss()
        self.mask_bce_loss = nn.BCEWithLogitsLoss()
        self.seg_ce_loss_weight = seg_ce_loss_weight
        self.mask_ce_loss_weight = mask_ce_loss_weight
        self.mask_dice_loss_weight = mask_dice_loss_weight
        self.score_loss_weight = score_loss_weight
        self.fts_loss_weight = fts_loss_weight
        self.triplet_loss_weight = triplet_loss_weight
        self.overlap_loss_weight = overlap_loss_weight
        self.triplet_margin = triplet_margin
        self.border_loss_weight = border_loss_weight
        self.equal_instance_weight = equal_instance_weight

        self.use_superpoint_pooling = use_superpoint_pooling
        # only relevant when use_superpoint_pooling=False: use the dataset's
        # seg_indices at eval to snap per-point masks to superpoints
        self.eval_superpoint_voting = eval_superpoint_voting

        self.positional_embedding = None
        if self.use_positional_encoding:
            self.positional_embedding = build_positional_embedding(positional_embedding)

    def query_pooling(self, data):
        queries = self.__query.weight[None, ...].repeat(len(data['offset']), 1, 1) 
        return queries
        """Permutation-equivariant query init: FPS-sampled features + positional encoding.

        Uses precomputed seed_ids (FPS indices in original point space) and maps
        them through seg_indices to get superpoint-level indices after pooling.
        """
        features = data['features']  # (N_pooled, C) after superpoint pooling
        offset = data['offset']      # (B,) cumulative counts in pooled space
        seed_ids = data['seed_ids']  # (B, num_seeds) indices in original point space

        # Map original-space seed indices to superpoint-space indices
        seg_indices = data['seg_indices']  # (N_original,) mapping original -> superpoint
        offset_orig = data['offset_orig'] # (B,) cumulative counts in original space

        seed_features_list = []
        seed_pos_list = []
        pos_enc = data.get('positional_embedding')  # (N_pooled, D) or None

        bs_orig = 0
        bs_pool = 0
        for i, (be_orig, be_pool) in enumerate(zip(offset_orig, offset)):
            # Map seed_ids from original space to pooled superpoint indices
            seed_sp = seg_indices[seed_ids[i] + bs_orig]  # superpoint indices (global)

            seed_features_list.append(features[seed_sp])
            if pos_enc is not None:
                seed_pos_list.append(pos_enc[seed_sp])

            bs_orig = be_orig
            bs_pool = be_pool

        seed_features = torch.stack(seed_features_list)          # (B, num_query, C)
        # L2-normalize to remove magnitude dependency (improves sim-to-real transfer)
        seed_features = torch.nn.functional.normalize(seed_features, dim=-1)
        query_features = self.query_proj(seed_features)          # (B, num_query, query_dim)

        # Add positional encoding at seed locations if available
        if pos_enc is not None:
            seed_pos = torch.stack(seed_pos_list)                # (B, num_query, D)
            query_features = query_features + seed_pos

        return query_features

    def __compute_loss(self, pred, data):
        auxiliary_losses = {
            'seg_ce': [],
            'mask_ce': [],
            'mask_dice': [],
            'matched_iou': [],
            'score_loss': [],
        }

        intersections = []
        unions = []

        for p in pred:
            indices = self.matcher(p, data)
            matched_outputs, matched_targets, matched_seg_outputs, matched_seg_targets, indices = \
                self.mask_selector(p, data, indices)

            # skip samples with empty matches — MaskSelector drops them too,
            # so the zip below stays aligned sample-by-sample
            matched_scores = [
                p['pred_score'][i][indices[i][0]][..., 0]
                for i in range(len(data['offset']))
                if indices[i][0] is not None and len(indices[i][0]) > 0
            ]

            t = {
                'seg_ce': [],
                'mask_ce': [],
                'mask_dice': [],
                'matched_iou': [],
                'score_loss': [],
            }

            for score, mask, target, p_seg, t_seg in zip(
                matched_scores, matched_outputs, matched_targets,
                matched_seg_outputs, matched_seg_targets,
            ):
                if self.seg_ce_loss_weight > 0:
                    t['seg_ce'].append(self.semantic_ce_loss(p_seg, t_seg))
                else:
                    t['seg_ce'].append(p_seg.new_tensor(0.0))

                if self.mask_ce_loss_weight > 0:
                    if self.equal_instance_weight:
                        # per-instance mean BCE (one value per matched mask),
                        # so each instance counts the same regardless of size
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(
                            mask, target.float(), reduction='none')
                        t['mask_ce'].append(bce.mean(0))  # (K,)
                    else:
                        t['mask_ce'].append(self.mask_bce_loss(mask, target.float()))
                else:
                    t['mask_ce'].append(mask.new_tensor(0.0))

                if self.mask_dice_loss_weight > 0:
                    if self.equal_instance_weight:
                        t['mask_dice'].append(self.__dice_per_instance(mask, target))  # (K,)
                    else:
                        t['mask_dice'].append(self.mask_dice_loss(mask, target))
                else:
                    t['mask_dice'].append(mask.new_tensor(0.0))

                with torch.no_grad():
                    intersections.append(((mask > 0) * target).sum(0))
                    unions.append(((mask > 0).sum(0) + target.sum(0)) - intersections[-1])
                    ious = intersections[-1] / unions[-1]
                    t['matched_iou'].append(ious.mean())

                if self.score_loss_weight > 0:
                    filter = ious > 0.5
                    if filter.sum() > 0:
                        t['score_loss'].append(torch.nn.functional.mse_loss(score[filter], ious[filter]))
                    else:
                        t['score_loss'].append(score.new_tensor(0.0))
                else:
                    t['score_loss'].append(score.new_tensor(0.0))

            for key in auxiliary_losses:
                if len(t[key]) > 0:
                    if self.equal_instance_weight and key in ('mask_ce', 'mask_dice'):
                        # flat-average over every instance in the batch so each
                        # instance contributes equally (not per-sample averaged)
                        auxiliary_losses[key].append(
                            torch.cat([v.reshape(-1) for v in t[key]]).mean())
                    else:
                        auxiliary_losses[key].append(torch.stack(t[key]).mean())

        for key in auxiliary_losses:
            if len(auxiliary_losses[key]) == 0:
                # whole batch had no valid instances
                auxiliary_losses[key] = torch.tensor(0.0, device=data['features'].device)
            elif key in ['matched_iou']:
                auxiliary_losses[key] = torch.stack(auxiliary_losses[key]).mean()
            elif key in ['mask_dice']:
                # Full weight on the final prediction, mean over the
                # auxiliary (refinement) layers.
                *aux, final = auxiliary_losses[key]
                auxiliary_losses[key] = (torch.stack(aux).mean() + final) if aux else final
            else:
                auxiliary_losses[key] = torch.stack(auxiliary_losses[key]).sum()

        auxiliary_losses['fts_loss'] = data['fts_loss']

        device = data['features'].device

        # Triplet loss: query-to-instance prototype
        if self.triplet_loss_weight > 0:
            auxiliary_losses['triplet_loss'] = self.__compute_triplet_loss(
                pred[-1], data
            )
        else:
            auxiliary_losses['triplet_loss'] = torch.tensor(0.0, device=device)

        # Overlap loss: penalise pairs of queries predicting the same region
        if self.overlap_loss_weight > 0:
            auxiliary_losses['overlap_loss'] = self.__compute_overlap_loss(
                pred[-1], data
            )
        else:
            auxiliary_losses['overlap_loss'] = torch.tensor(0.0, device=device)

        # Border loss: per-point binary border vs non-border classification.
        # Points without instance labels carry a negative border label
        # (GenerateBoundary) and are excluded from supervision.
        if self.border_loss_weight > 0 and 'border_logits' in data:
            border_valid = data['border'] >= 0
            if border_valid.any():
                auxiliary_losses['border_loss'] = self.border_loss(
                    data['border_logits'][border_valid],
                    data['border'][border_valid].float(),
                )
            else:
                auxiliary_losses['border_loss'] = torch.tensor(0.0, device=device)
        else:
            auxiliary_losses['border_loss'] = torch.tensor(0.0, device=device)

        auxiliary_losses['loss'] = (
            self.seg_ce_loss_weight * auxiliary_losses['seg_ce']
            + self.mask_ce_loss_weight * auxiliary_losses['mask_ce']
            + self.mask_dice_loss_weight * auxiliary_losses['mask_dice']
            + self.score_loss_weight * auxiliary_losses['score_loss']
            + self.fts_loss_weight * auxiliary_losses['fts_loss']
            + self.triplet_loss_weight * auxiliary_losses['triplet_loss']
            + self.overlap_loss_weight * auxiliary_losses['overlap_loss']
            + self.border_loss_weight * auxiliary_losses['border_loss']
        )

        # Drop disabled (weight 0) loss terms so they don't clutter the logs.
        # The total 'loss' and the non-weighted 'matched_iou' metric are kept.
        term_weights = {
            'seg_ce': self.seg_ce_loss_weight,
            'mask_ce': self.mask_ce_loss_weight,
            'mask_dice': self.mask_dice_loss_weight,
            'score_loss': self.score_loss_weight,
            'fts_loss': self.fts_loss_weight,
            'triplet_loss': self.triplet_loss_weight,
            'overlap_loss': self.overlap_loss_weight,
            'border_loss': self.border_loss_weight,
        }
        for key, weight in term_weights.items():
            if weight == 0:
                auxiliary_losses.pop(key, None)

        return auxiliary_losses

    def __dice_per_instance(self, mask, target):
        """Per-instance (per-column) dice loss, matching DiceLoss but without
        the final mean over instances — returns a (K,) vector so each matched
        mask can be weighted equally downstream."""
        smooth = self.mask_dice_loss.smooth
        pred = mask.sigmoid()
        numerator = 2 * (pred * target).sum(0) + smooth
        denominator = pred.sum(0) + target.sum(0) + smooth
        return self.mask_dice_loss.loss_weight * (1 - numerator / denominator)

    def __compute_overlap_loss(self, last_pred, data):
        """Penalise pairs of queries whose predicted masks significantly overlap.

        Uses soft pairwise IoU over sigmoid probabilities so the loss is
        differentiable and pushes queries toward non-overlapping regions.
        """
        masks = last_pred['pred_masks'].sigmoid()  # (N, num_query)
        offset = data['offset']
        losses = []
        bs = 0
        for be in offset:
            m = masks[bs:be].T.float()       # (Q, N_i)
            inter = m @ m.T                  # (Q, Q)  — soft intersection
            areas = m.sum(dim=1)             # (Q,)
            union = areas[:, None] + areas[None, :] - inter  # (Q, Q)
            soft_iou = inter / (union + 1e-6)
            # mask out diagonal (self-overlap is always 1, not penalised)
            eye = torch.eye(soft_iou.shape[0], device=soft_iou.device)
            off_diag_iou = soft_iou * (1.0 - eye)
            losses.append(off_diag_iou.mean())
            bs = be
        return torch.stack(losses).mean()

    def __compute_triplet_loss(self, last_pred, data):
        """Query-to-instance-prototype triplet loss.

        For each matched query, the anchor is the query embedding,
        the positive is the mean feature of the matched GT instance,
        and the negative is the mean feature of the hardest other
        GT instance (closest prototype to the anchor).
        """
        query_feat = last_pred.get('query_feat')  # (B, num_query, D)
        if query_feat is None:
            return torch.tensor(0.0, device=data['features'].device)

        # Use projected point features (same dim as queries) for prototypes
        point_feat = last_pred.get('query_point_features')  # (N_pooled, D)
        if point_feat is None:
            return torch.tensor(0.0, device=data['features'].device)

        instances = data['instance']  # (N_pooled,) GT instance labels
        offset = data['offset']       # (B,)

        # Run matcher on last prediction to get query-to-GT assignment
        indices = self.matcher(last_pred, data)

        all_losses = []
        bs = 0
        for i, be in enumerate(offset):
            pred_ids, tgt_ids = indices[i]
            if pred_ids is None or len(pred_ids) == 0:
                bs = be
                continue

            inst = instances[bs:be]
            feat = point_feat[bs:be]  # (N_i, D) same dim as query features

            # Build prototype for each GT instance in this sample
            unique_inst = inst.unique()
            unique_inst = unique_inst[unique_inst != -1]
            if unique_inst.numel() < 2:
                bs = be
                continue

            # (K, C) one prototype per GT instance
            prototypes = torch.stack([
                feat[inst == uid].mean(dim=0) for uid in unique_inst
            ])
            prototypes = F.normalize(prototypes, dim=-1)

            # Map tgt_ids (matched GT instance indices) to unique_inst ordering
            inst_id_to_idx = {uid.item(): idx for idx, uid in enumerate(unique_inst)}

            for qi, ti in zip(pred_ids, tgt_ids):
                anchor = F.normalize(query_feat[i, qi].unsqueeze(0), dim=-1)  # (1, D)

                # ti is the GT instance index (0-based after superpoint pooling remapping)
                if ti.item() not in inst_id_to_idx:
                    continue
                pos_idx = inst_id_to_idx[ti.item()]
                positive = prototypes[pos_idx].unsqueeze(0)  # (1, D)

                # Hard negative: closest prototype that is not the positive
                neg_mask = torch.ones(prototypes.shape[0], dtype=torch.bool,
                                      device=prototypes.device)
                neg_mask[pos_idx] = False
                neg_protos = prototypes[neg_mask]  # (K-1, D)
                sims = (anchor @ neg_protos.T).squeeze(0)  # (K-1,)
                hard_neg_idx = sims.argmax()
                negative = neg_protos[hard_neg_idx].unsqueeze(0)  # (1, D)

                # Triplet: d(anchor, positive) - d(anchor, negative) + margin
                d_pos = 1.0 - (anchor * positive).sum()
                d_neg = 1.0 - (anchor * negative).sum()
                loss = torch.clamp(d_pos - d_neg + self.triplet_margin, min=0.0)
                all_losses.append(loss)

            bs = be

        if len(all_losses) == 0:
            return point_feat.new_tensor(0.0)
        
        return torch.stack(all_losses).mean()

    def __get_pos_encs(self, data_dict):
        if self.positional_embedding is None:
            data_dict['positional_embedding'] = None
            return data_dict

        pos_encodings_pcd = []
        bs = 0
        with autocast("cuda", enabled=False):
            for be in data_dict['offset']:
                coords = data_dict['coord'][bs:be]
                scene_min = coords.min(dim=0)[0][None, ...]
                scene_max = coords.max(dim=0)[0][None, ...]
                tmp = self.positional_embedding(
                    coords[None, ...].float(), input_range=[scene_min, scene_max],
                )
                pos_encodings_pcd.append(tmp.squeeze(0).permute((1, 0)))
                bs = be

        data_dict['positional_embedding'] = torch.cat(pos_encodings_pcd)
        return data_dict

    def forward(self, data):
        data.update(self.encoder(data))
        data.update(self.__get_pos_encs(data))

        # Border branch: predict on the original-resolution per-point features,
        # before superpoint pooling collapses them. 'border' GT is not a pooling
        # key, so both logits and labels stay at original point resolution.
        if self.border_head is not None:
            data['border_logits'] = self.border_head(data['features']).squeeze(-1)

        # when superpoint pooling is off, the model must stay per-point, so
        # seg_indices are removed from data — but we keep them aside so
        # select_masks can still snap per-point masks to superpoints at eval
        eval_superpoints = None
        if not self.use_superpoint_pooling:
            eval_superpoints = data.pop('seg_indices', None)
            if not self.eval_superpoint_voting:
                eval_superpoints = None

        data = self.superpoint_pooling(data, ['instance', 'segment', 'features'])

        queries = self.query_pooling(data)

        pred = self.decoder(data, queries)

        # Loss/matcher must run in fp32 under AMP: the Hungarian matcher's cost
        # einsums and the mask reductions sum over thousands of points and
        # overflow to inf/NaN in fp16 (linear_sum_assignment then rejects the
        # cost matrix). Upcast the prediction tensors and disable autocast for
        # the whole loss computation. No-op when AMP is off (already fp32); the
        # original fp16 `pred` is kept for the eval select_masks branch below.
        with torch.cuda.amp.autocast(enabled=False):
            pred_fp32 = [
                {
                    k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                    for k, v in p.items()
                }
                for p in pred
            ]
            return_dict = self.__compute_loss(pred_fp32, data)

        if not self.training:
            superpoints = eval_superpoints if eval_superpoints is not None else data['seg_indices']
            return_dict.update(select_masks(pred[-1], superpoints.cpu(), coord=data['coord'], **self.mask_selection))
            if self.border_head is not None:
                return_dict['pred_border'] = data['border_logits'].sigmoid()
            data = self.superpoint_unpooling(data)

        return return_dict
