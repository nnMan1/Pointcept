_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16 # bs: total bs in all gpus
num_worker = 32
mix_prob = 0
empty_cache = True
enable_amp = False
evaluate = True
# resume=True
weight='/home/exp/abc_dataset_hungarian_matcher/insseg-mask3d-v1m1-0-spunet-base_dice_loss+focall_loss/model/model_last.pth'

class_names = [
    "assembly",
]
num_classes = 1
segment_ignore_index = (-1, )

# model settings
model = dict(
    type="Mask-3D",
    backbone=dict(
        type="MinkUNet34C",
        in_channels = 3,
        out_channels = 128,
        out_fpn=True, #return intermidiate features
    ),
    position_encoding=dict(
        type='PositionEmbeddingCoordsSine',
        pos_type="fourier",
        d_pos=128,
        gauss_scale=1,
        normalize=True,
    ),
    mask_module_config=dict(
        num_classes=1, 
        return_attn_masks=True, 
        use_seg_masks=False
    ),
    query_refinement_config=dict(
        pre_norm=False,
        num_heads=8, 
        dropout=0
    ),
    num_decoders=1,
    dim_feedforward=1024,
    hidden_dim=128,
    mask_dim=128
)

# scheduler settings
epoch = 800
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.2)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "Assembly"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="NormalizeCoord"),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.8),
            dict(type="RandomJitter", sigma=0.001, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[2, 4], [8, 16]]),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment", "instance"),
            ),
            # dict(type="SphereCrop", sample_rate=0.8, mode="random"),
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
                    "instance_centroid",
                    "bbox",
                    "seed_ids",
                    "id",
                    "path"
                ),
                feat_keys=("grid_coord"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeCoord"),
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
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment", "instance"),
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
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "seed_ids"
                ),
                feat_keys=('coord'),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(),  # currently not available
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="MyInsSegEvaluator",
    ),
    dict(type="CheckpointSaver", save_freq=None),
]
