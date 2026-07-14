_base_ = ["../_base_/default_runtime.py"]

# EQUAL-INSTANCE-WEIGHT variant of the H5-dataset dinocache config.
# Identical to insseg-...-VI-dinocache-h5 except equal_instance_weight=True:
# the mask BCE/dice losses are averaged per-instance (one value per matched
# mask) and then flat-averaged across all instances in the batch, so every
# instance contributes equally regardless of its point count. The default
# (False) sums BCE / area-weights dice, which lets large instances dominate.
#
# H5-DATASET variant of insseg-...-VI-border-dinocache.
# Same model/training as the dinocache config, but train/val read the PACKED
# abc_dataset HDF5 shards (HDF5_Dataset) instead of MechanicalAssemblySynth,
# and the precomputed DinoV3 grids come from the abc_dataset feature cache
# written by configs/abc_dataset/extract-dino-v3-features.py.
#
# NOTE (must resolve before training): HDF5_Dataset.get_data does NOT emit
# 'seg_indices' (the abc HDF5 shards carry no superpoint clustering), but the
# GridSample keys and the model's select_masks require it. Either add a
# seg_indices field to HDF5_Dataset.get_data or compute a clustering from the
# stored mesh (mesh_vertices/mesh_faces) at pack time. See the chat notes.
#
# Requires /leonardo_scratch bound inside the container (see
# scripts/start_feature_extraction.sh for the --bind flags).

# misc custom setting
batch_size = 8 # bs: total bs in all gpus
num_worker = 32
mix_prob = 0
empty_cache = True
enable_amp = False
evaluate = True
sync_bn = True
find_unused_parameters = True

# large fp16 feature grids are streamed per-sample by the model
keep_on_cpu = ("image_features",)

# synthetic features now come from the abc_dataset HDF5 extraction
features_root_synth = "/leonardo_scratch/large/userexternal/vdosljak/dino_features/abc_dataset/features"
features_root_real = "/leonardo_scratch/large/userexternal/vdosljak/dino_features/cetim_real/features"

num_classes = 1
fts_sizes = 128
dim_feedforward = 1024
instance_ignore_index = -1
segment_ignore_index = (-1, )

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
            freeze_backbone_bn=False
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
            return_features=['feat']
            ),
        ),
        backbone_out_channels=64,
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
    # Weight every matched instance equally in mask_ce/mask_dice, independent
    # of its point count (otherwise large instances dominate the mask losses).
    equal_instance_weight=True,
    # Border detection branch: per-point binary border vs non-border head.
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
# train/val now read the packed abc_dataset HDF5 shards; test stays on the
# raw cetim real assemblies (MechanicalAssemblyV2).
hdf5_data_root = "data/segment-assembly-synthetic/data/abc_dataset/processed"
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
        type="HDF5_Dataset",
        split="train",
        data_root=hdf5_data_root,
        # load_images builds mappings_src/tgt from the stored views (the model
        # scatters the cached grids onto points via these mappings); HDF5_Dataset
        # has no separate "mappings-only" flag, so PNGs are decoded too.
        load_images=True,
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
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.8),
            dict(type="RandomJitter", sigma=0.001, clip=0.02),
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
        type="HDF5_Dataset",
        split="val",
        data_root=hdf5_data_root,
        load_images=True,
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
    instance_ignore_index=instance_ignore_index,
    verbose=False,
)
