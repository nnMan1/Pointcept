_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 30  # bs: total bs in all gpus
num_worker = 30
mix_prob = 0
empty_cache = True
enable_amp = True
evaluate = True
# resume=True
# weight='backbones/sstnet_pretrain.pth'
# weight='exp/fuselage_instance/insseg-spunet-v1m1-0-spformer-res-base/model/model_last.pth'

classes={"other": 0, 
         "gear": -1, 
         "nut": -1, 
         "screw": -1, 
         "axe": -1, 
         "rivet": 1, 
         "sting-stif": 2, 
         "ruber-seal": 3, 
         "main_panel": 4,
         "hole": 5,
         "rivet_t1": 1,
         "rrivet_t2": 1}

class_names = ["other", "rivet", "string-stif", "ruber-seal", "main-panel", "hole"]


fts_sizes = 128
dim_feedforward=512
segment_ignore_index = (-1, )
num_classes = 6
segment_ignore_index = (-1, )

model = dict(
    type="GroupingSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 64, 128, 128, 96, 256),
        layers=(2, 3, 4, 2, 2, 2),
    ),
    final_in_channels = 256,
    num_classes=6,
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(type="FocalLoss", loss_weight=1.0, ignore_index=-1)
    ],
)


# scheduler settings
epoch = 500
eval_epoch = 100  # sche total eval & checkpoint epoch
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
dataset_type = "MechanicalAssembly"
data_root = "data/Fuselage/crops"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.1),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", "seg_indices"),
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
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
                    "seg_indices"
                ),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
        classes=classes
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
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
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", "seg_indices"),
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
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
                    "seg_indices"
                ),
                feat_keys=("coord", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
        classes=classes
    ),
    test=dict(),  # currently not available
)

# hooks = [
#     dict(type="CheckpointLoader", keywords="module.", replacement="module."),
#     # dict(type="CheckpointLoader", keywords="module.", replacement="module.encoder.backbone."),
#     dict(type="IterationTimer", warmup_iter=2),
#     dict(type="InformationWriter"),
#     dict(
#         type="InsSegEvaluator",
#         segment_ignore_index=segment_ignore_index,
#         instance_ignore_index=-1,
#     ),
#     dict(type="CheckpointSaver", save_freq=None),
# ]
