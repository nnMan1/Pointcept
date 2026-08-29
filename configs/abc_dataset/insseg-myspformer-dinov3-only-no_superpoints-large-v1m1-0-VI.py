_base_ = ["../_base_/default_runtime.py"]

# DinoV3-only with PRECOMPUTED DinoV3 features (no ViT at train time).
# Identical to insseg-...-VI except:
#   * Image2PointCLoud has model_type=None: the 840M-param DinoV3 is not
#     instantiated; per-view patch grids come from HDF5 shards written by
#     scripts/start_feature_extraction.sh (run it first!).
#   * LoadImageFeatures transform attaches the fp16 grids; Collect ships
#     'image_features' instead of 'images'.
#   * keep_on_cpu: the grids stay in pinned host memory; the model streams
#     one sample's views to GPU at a time.
# Requires /leonardo_scratch bound inside the container (see
# scripts/start_feature_extraction.sh for the --bind flags).

# misc custom setting
batch_size = 4 # bs: total bs in all gpus
num_worker = 32
mix_prob = 0
empty_cache = True
enable_amp = False
evaluate = True
sync_bn = True
find_unused_parameters = True
# weight = 'exp/abc_dataset/insseg-myspformer_ptv3-v1m1-0-spunet-base/model/model_last.pth'
# resume = True# weight='backbones/sstnet_pretrain.pth'

# large fp16 feature grids are streamed per-sample by the model
keep_on_cpu = ("image_features",)

features_root_synth = "/leonardo_scratch/large/userexternal/vdosljak/dino_features/mech_synth/features"
features_root_real = "/leonardo_scratch/large/userexternal/vdosljak/dino_features/cetim_real/features"

num_classes = 1
fts_sizes = 128
dim_feedforward=1024
segment_ignore_index = (-1, )
instance_ignore_index = -1

# DinoV3 image feature dim projected onto points (Image2PointCLoud.out_fts_dim)
dinov3_out_dim = 256

model = dict(
    type="MySPFormer",
    num_query = 100,
    encoder=dict(
        # DinoV3-only backbone: project image features onto points, no PT-V3.
        backbone=dict(
            type="Image2PointCLoud",
            model_type=None,  # precomputed features: no image backbone
            model_name=None,
            fts_dim=1280,
            out_fts_dim=dinov3_out_dim,
            merge_strategy='random_sample',
            return_features=['feat'],
            freeze_backbone=False,
            freeze_backbone_bn=False
        ),
        backbone_out_channels=dinov3_out_dim,
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
    # Positional encoding: toggle with use_positional_encoding (True/False).
    # When on, the per-point PE is added to the cross-attention keys AND fused
    # into the mask features (position-aware mask prediction), which helps the
    # image-only model separate identical-appearance parts.
    use_positional_encoding=True,
    positional_embedding=dict(
        type='PositionEmbeddingCoordsSine',
        pos_type='fourier',
        d_pos=128,
        d_in=3,
        gauss_scale=1.0,
        normalize=True,
        scale=6.2832),
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
recompute_clustering=False
image_size=(512, 512)

classes={"other": 0,
        "gear": 0,
        "nut": 0,
        "screw": 0,
        "axe": 0}

class_names = ["other"]


data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=['class_names'],
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
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.5),
            # dict(type="NormalizeCoord"),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.8),
            dict(type="RandomJitter", sigma=0.001, clip=0.02),
            # dict(type='ElasticDistortion', distortion_params=[[20, 40], [80, 160]]),
            dict(
                type="GridSample",
                grid_size=1,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", 'seg_indices'),
            ),
            # dict(type="SphereCrop",  point_max=200000, mode="random"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type='RandomSeed', n_points = 100),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "image_features",
                    "mappings_src",
                    "mappings_tgt",
                    # "instance_centroid",
                    # "bbox",
                    "seed_ids",
                    "seg_indices",
                    "instance_segment",
                    "path",
                    "name",
                    "inverse"
                ),
                feat_keys=("coord", "normal"),
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
            # dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
               grid_size=1,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", "seg_indices"),
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type='FPSSeed', n_points = 100),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "image_features",
                    "mappings_src",
                    "mappings_tgt",
                    'origin_coord', 'origin_segment', 'origin_instance',
                    # "instance_centroid",
                    # "bbox",
                    "instance_segment",
                    "seed_ids",
                    "seg_indices",
                    "path",
                    "name",
                    "inverse"
                ),
                feat_keys=("coord", "normal"),
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
            dict(type='FPSSeed', n_points=100),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                # [Change: added seed_ids to test Collect keys]
                keys=('coord', 'face', 'grid_coord', 'segment', 'instance',
                      'instance_segment', 'image_features', 'mappings_src',
                      'mappings_tgt', 'origin_coord', 'origin_segment',
                      'origin_instance', 'seed_ids', 'seg_indices', 'path',
                      'name', 'inverse', 'instance_segment'),
                feat_keys=('coord', 'normal'),
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
    # dict(type="CheckpointLoader", keywords=["module.", "decoder.mask_modules.0.class_embed_head"], replacement=["module.", "dummy."]),
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
