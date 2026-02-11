_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 32
num_worker = 32
mix_prob = 0
empty_cache = True
enable_amp = True
evaluate = True
prefetch_factor=2

num_classes = 1
fts_sizes = 128
dim_feedforward=1024
segment_ignore_index = (-1, )

# model settings
model = dict(
    type="FeatureDistiller",
    backbone_student=dict(
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
        ),
    # backbone_teacher=dict(
    #     type='Image2PointCLoud',
    #     return_features = ['feat'],
    #     project_fts = False,
    #     merge_strategy="mean",
    #     model_type = None,
    #     model_name = "DinoV2",
    #     fts_dim=1280
    # ),
    student_out_channels=64,
    teacher_out_channels=384,
    criteria=[
        dict(type="MSELoss", loss_weight=1.0),
        dict(type="CosineEmbeddingLoss", loss_weight=1.0)
    ],
    project_fts=True,
)

epoch = 100
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.000)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)


classes=dict({
            'other': 0,
            'nut': 0,
            'screw': 0
        })

class_names = ["other"]

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type='HDF5_Dataset',
        split='train',
        data_root='data/data/processed/abc_dataset',
        load_images=True,
        # image_size=(512, 512),
        load_features=dict(
            image_features='preextracted_features/dino_v2_small/features_train'
        ),
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
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.8),
            dict(type="RandomJitter", sigma=0.001, clip=0.02),
            dict(
                type="GridSample",
                grid_size=2,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance", 'image_features'),
            ),
            dict(
                type="Copy",
                keys_dict={
                    "image_features": "feat_teacher"
                }
            ),
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
                    # "instance",
                    # "instance_segment",
                    # "images",
                    # "mappings_src",
                    # "mappings_tgt",
                    "path",
                    "name",
                    "inverse",
                    "name",
                    "feat_teacher"
                ),
                feat_keys=("coord"),
                offset_keys_dict=dict(
                    offset="coord", 
                    origin_offset="origin_coord", 
                    mappings_offset="mappings_src",
                    instance_segment_offset="instance_segment"
                ),
            ),
        ],
        test_mode=False,
        classes=classes,
    ),
    val=dict(
        type='HDF5_Dataset',
        split='train',
        data_root='data/data/processed/abc_dataset',
        load_images=True,
        # image_size=(512, 512),
        load_features=dict(
            image_features='preextracted_features/dino_v2_small/features_train'
        ),
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
                grid_size=2,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance"'image_features'),
            ),
            dict(
                type="Copy",
                keys_dict={
                    "image_features": "feat_teacher"
                }
            ),
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
                    # "instance",
                    # "instance_segment",
                    # "mappings_src",
                    # "mappings_tgt",
                    "path",
                    "name",
                    "inverse",
                    "name",
                    "feat_teacher"
                ),
                feat_keys=("coord"),
                offset_keys_dict=dict(
                    offset="coord", 
                    origin_offset="origin_coord", 
                    mappings_offset="mappings_src",
                    instance_segment_offset="instance_segment"
                ),
            ),
        ],
        test_mode=False,
        classes=classes,
    ))

