_base_ = ["../_base_/default_runtime.py"]

# SCALE-ROBUST variant of concat-amp-norm: identical config (same merged
# data, model, schedule, from scratch — NO pretrained init) plus
# RandomRescaleDiag(80..480) in the train transform. Isolates the effect of
# scale-robust training: the load-time pow2 hack leaves scenes at arbitrary
# power-of-2 scales in [80,400] while CETIM test scans are unnormalized
# (diag 77-467); rescaling each scene to a random diagonal per epoch makes
# the model robust across that whole range. Compare directly against
# insseg-...-VI-dinocache-concat-amp-norm (real AP50 0.2388 / mIoU 0.3812).

#
# ELASTIC variant: adds ElasticDistortion([[4, 8], [16, 32]]) on top of the
# rescale recipe — smooth local geometric warping as scan-realism
# augmentation (real scans deviate from CAD-perfect surfaces at ~mm scales).
# Placed AFTER RandomRescaleDiag so magnitudes live in the rescaled units,
# with the other geometric augs. Params are [granularity, magnitude] in
# coordinate units (~mm): gentle 4mm- and 16mm-scale warps.

# misc custom setting
batch_size = 8 # bs: total bs in all gpus
num_worker = 32
mix_prob = 0
empty_cache = True
enable_amp = True
evaluate = True
sync_bn = True
find_unused_parameters = True

# large fp16 feature grids are streamed per-sample by the model
keep_on_cpu = ("image_features",)

features_root_synth = "/leonardo_scratch/large/userexternal/vdosljak/dino_features/mech_synth/features"
features_root_real = "/leonardo_scratch/large/userexternal/vdosljak/dino_features/cetim_real/features"

num_classes = 1
fts_sizes = 128
dim_feedforward = 1024
segment_ignore_index = (-1, )
instance_ignore_index = -1

# Border detection branch settings
border_radius = 4.0  # metric radius for boundary generation (~4x grid_size)

model = dict(
    type="MySPFormer",
    num_query = 100,
    encoder=dict(
        backbone=dict(
        type="MergeFeatures",
        model1_config=dict(
            type="Image2PointCLoud",
            model_type=None,  # precomputed features: no image backbone
            model_name=None,
            fts_dim=1280,
            out_fts_dim=256,
            merge_strategy='random_sample',
            return_features=['feat'],
            freeze_backbone=False,
            freeze_backbone_bn=False,
            # trailing LayerNorm on the projection: scale-control the DinoV3
            # branch at its source (helps fusion balance + synth->real transfer)
            proj_norm=True,
        ),
        model2_config=dict(
            type="PT-V3FeatureExtractor",
            in_channels=3,
            order=["z", "z-trans", "hilbert", "hilbert-trans"],
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
            pdnorm_decouple=True,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
            return_features=['feat'],
            # concat the DinoV3 features onto the final decoder output instead
            # of averaging them in per level (default 'average').
            fusion='concat',
            # LayerNorm each branch before the concat so neither dominates the
            # mask-feature head input.
            concat_norm=True,
            ),
        ),
        # 64 (PT-V3 decoder output) + 256 (out_fts_dim of the DinoV3 branch)
        backbone_out_channels=64 + 256,
        out_channels=32
     ),
    decoder=dict(
        in_channels=32,
        hlevels=6,
        mask_modules=[
            dict(
                num_classes=num_classes,
                return_attn_masks=True,
                hidden_dim=fts_sizes,
                reuse=1
            )
        ],
        query_refinement_modules=[
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8,
                dropout=0.005
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8,
                dropout=0.005
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8,
                dropout=0.005
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8,
                dropout=0.005
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8,
                dropout=0.005
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8,
                dropout=0.005
            )
        ],
    ),
    matcher=dict(
        type='HungarianMatcher',
        cost_terms=[
            dict(type='ClassCost', weight=0.5, enabled=True, use_logits=False),
            dict(type='MaskBCECost', weight=1.0, enabled=True, instance_ignore_index=-1),
            dict(type='MaskDiceCost', weight=1.0, enabled=True, instance_ignore_index=-1)
        ],
        instance_ignore_index=-1
    ),
    use_superpoint_pooling=False,
    mask_dice_loss_weight=7.0,
    # Border detection branch: per-point binary border vs non-border head.
    # Off (matches the concat-amp-norm model); border labels still computed by
    # GenerateBoundary but unused at weight 0.
    border_loss_weight=0.0,
    border_focal_alpha=0.5,
)

# scheduler settings
epoch = 500
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "MechanicalAssemblySynth"
data_root = "data/segment-assembly-merged-synthetic/data"
recompute_clustering = False
image_size = (512, 512)

classes={"other": 0,
        "gear": 0,
        "nut": 0,
        "screw": 0,
        "axe": 0}

class_names = ["other"]


data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        cache=True,
        data_root=data_root,
        recompute_clustering=recompute_clustering,
        image_size=image_size,
        transform=[
            dict(type="LoadImageFeatures", features_root=features_root_synth),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRescaleDiag", lo=80, hi=480),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.8),
            dict(type="RandomJitter", sigma=0.001, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[4, 8], [16, 32]]),
            dict(
                type="GridSample",
                grid_size=1,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", 'seg_indices'),
            ),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            # Border label on the grid-sampled cloud: bounded density -> cheap
            # radius graph, and aligned with the points the model predicts on.
            dict(type="GenerateBoundary", radius=border_radius),
            dict(type='RandomSeed', n_points = 100),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "border",
                    "image_features",
                    "mappings_src",
                    "mappings_tgt",
                    "seed_ids",
                    "seg_indices",
                    "instance_segment",
                    "path",
                    "name",
                    "inverse"
                ),
                feat_keys=("coord",),
                offset_keys_dict=dict(
                    offset="coord",
                    origin_offset="origin_coord",
                    image_offset="image_features",
                    mappings_offset="mappings_src",
                    instance_segment_offset="instance_segment"),
            ),
        ],
        test_mode=False,
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        cache=True,
        recompute_clustering=recompute_clustering,
        image_size=image_size,
        transform=[
            dict(type="LoadImageFeatures", features_root=features_root_synth),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=1,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", "seg_indices"),
            ),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="GenerateBoundary", radius=border_radius),
            dict(type='FPSSeed', n_points = 100),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "border",
                    "image_features",
                    "mappings_src",
                    "mappings_tgt",
                    'origin_coord', 'origin_segment', 'origin_instance',
                    "instance_segment",
                    "seed_ids",
                    "seg_indices",
                    "path",
                    "name",
                    "inverse"
                ),
                feat_keys=("coord",),
                offset_keys_dict=dict(
                    offset="coord",
                    origin_offset="origin_coord",
                    image_offset="image_features",
                    mappings_offset="mappings_src",
                    instance_segment_offset="instance_segment"),
            ),
        ],
        test_mode=False,
        classes=classes,
    ),
        test=dict(
        type='MechanicalAssemblyV2',
        split='all',
        image_size=image_size,
        data_root='/home/data/cetim_assembly/dataset/downsampled1',
        load_image_files=False,  # mappings only; features come from the cache
        transform=[
            dict(type="LoadImageFeatures", features_root=features_root_real),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    segment='origin_segment',
                    instance='origin_instance')),
            dict(
                type='GridSample',
                grid_size=1,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True,
                keys=('coord', 'normal', 'segment', 'instance', 'seg_indices')),
            dict(type='CenterShift', apply_z=False),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='GenerateBoundary', radius=border_radius),
            dict(type='FPSSeed', n_points=100),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'face', 'grid_coord', 'segment', 'instance', 'border',
                      'instance_segment', 'image_features', 'mappings_src',
                      'mappings_tgt', 'origin_coord', 'origin_segment',
                      'origin_instance', 'seed_ids', 'seg_indices', 'path',
                      'name', 'inverse'),
                feat_keys=('coord',),
                offset_keys_dict=dict(
                    offset='coord',
                    origin_offset='origin_coord',
                    image_offset='image_features',
                    mappings_offset='mappings_src',
                    instance_segment_offset='instance_segment'))
        ],
        test_mode=False,
        classes=dict(other=0, gear=0, nut=0, screw=0, axe=0)))

hooks = [
    dict(type="CheckpointLoader", keywords=["module."], replacement=["module."]),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator",
         segment_ignore_index=segment_ignore_index,
         instance_ignore_index=-1,),
    dict(type="CheckpointSaver", save_freq=None),
]

# Tester
test = dict(
    type="InstSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=False,
)
