_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8 # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = True
enable_amp = False
# resume=True
# weight='exp/fuselage_lr_split/semseg-pt-v3-0-base-ce-loss/model/model_last.pth'


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

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 4),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(48, 48, 48, 48),
        dec_depths=(2, 2, 2),
        dec_channels=(64, 64, 128),
        dec_num_head=(4, 4, 8),
        dec_patch_size=(48, 48, 48),
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
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 400
eval_epoch = 100# sche total eval & checkpoint epoch
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings

# dataset settings
dataset_type = "MechanicalAssembly"
data_root = "data/fuselage/crops"
num_classes = 6

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        augment_holes=True,
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
            dict(
                type="GridSample",
                grid_size=0.3,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "normal", "segment"),
            ),
            # dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
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
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.3,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "normal", "segment"),
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
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
#     dict(type="CheckpointLoader", keywords=["module.backbone.final"], replacement=["module.dummy"]),
#     dict(type="IterationTimer", warmup_iter=2),
#     dict(type="InformationWriter"),
#     dict(type="SemSegEvaluator"),
#     dict(type="CheckpointSaver", save_freq=None),
#     dict(type="PreciseEvaluator", test_last=False),
# ]
