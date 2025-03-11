_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4 # bs: total bs in all gpus
mix_prob = 0
empty_cache = True
enable_amp = False
resume=True
weight='exp/assembly/semseg-spunet-v1m1-0-base/model/model_best.pth'

# model settings
model = dict(
    type="EdgesDetector",
     backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=1,
        channels=(32, 64, 128, 128, 96, 96),
        layers=(2, 3, 4, 2, 2, 2),
    ),
    criteria=[
        dict(type="BCELoss", loss_weight=1.0)
    ],
)

# scheduler settings
epoch = 400
eval_epoch = 100# sche total eval & checkpoint epoch
optimizer = dict(type="SGD", lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "Assembly"
data_root = 'data/assembly'

data = dict(
    num_classes=1,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout", dropout_ratio=0.5, dropout_application_ratio=0.2,
                        keys=("coord", "grid_coord" "segment", "instance", "border_dist"),
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 16, 1/16], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 16, 1/16], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(type="RBFunction", key="border_dist", gamma=0.3),
                    dict(
                        type="GridSample",
                        grid_size=0.3,
                        hash_type="fnv",
                        mode="train",
                        keys=("coord", "segment", "instance", "border_dist"),
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=100000, mode="random",
                            keys=("coord", "grid_coord", "segment", "instance", "border_dist"),
                            ),
                    dict(type="CenterShift", apply_z=False),
                    # dict(type="NormalizeColor"),
                    # dict(type="ShufflePoint"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("grid_coord", "coord", "segment","instance", "border_dist"),
                        feat_keys=("coord", ),
                    ),
                ],
        multiview=5,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
                    dict(type="CenterShift", apply_z=True),   
                    # dict(type="RBFunction", key="border_dist", gamma=0.3),
                    dict(type='ClipFeature', key='border_dist', min_value=0, max_value=2),
                    dict(type='ScaleFeature', key='border_dist', min_value=2, max_value=0),
                    dict(
                        type="GridSample",
                        grid_size=0.3,
                        hash_type="fnv",
                        mode="train",
                        keys=("coord", "segment", "instance", "border_dist"),
                        return_grid_coord=True,
                    ),
                    dict(type="CenterShift", apply_z=False),
                    # dict(type="NormalizeColor"),
                    # dict(type="ShufflePoint"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("grid_coord", "coord", "segment","instance", "border_dist"),
                        feat_keys=("coord", ),
                    ),
                ],
                multiview=5,
                test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.3,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,                
                keys=("coord", "segment"),
            ),
            crop=None,
            post_transform=[
                dict(type="RBFunction", key="border_dist", gamma=0.3),
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("grid_coord",),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
            multiview=5,
        ),
    ),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="EdgeDetectionEvaluator",),
    dict(type="CheckpointSaver", save_freq=None),
]
