_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4 # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = True
enable_amp = False
# resume=True
# weight='/home/exp/fuselage/semseg-pt-v1-0-base_250x250x250_hard_rot_uniform_2/model/model_best.pth'

num_classes = 3
# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=3,
    backbone_out_channels=64,
    backbone=dict(
        type="MergeFeatures",
        model1_config=dict(
            type="DinoV2FeatureExtractor",
            fts_dim=384,
            merge_strategy='random_sample',
            return_features=['feat'],
            freeze_backbone=True,
            freeze_backbone_bn=True
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
        ),
    ),
    criteria=[dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
              dict(type='FocalLoss', loss_weight=1.0, ignore_index=-1)],
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
dataset_type='MechanicalAssemblyV2'
data_root='data/cetim_assembly/dataset/downsampled'

classes={"other": 0, 
        "gear": 0, 
        "nut": 1, 
        "screw": 2, 
        "axe": 0}

class_names = ["other", "nut", "screw"]


data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        cache=False,
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(
            #     type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            # ),
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
            # dict(type="ElasticDistortion", distortion_params=[[2, 4], [8, 16]]),
            dict(
                type="GridSample",
                grid_size=0.3,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance"),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "images",
                    "mappings_src",
                    "mappings_tgt",
                    "seg_indices",
                    "path",
                    "name",
                    "inverse"
                ),
                feat_keys=("coord"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord", image_offset="images", mappings_offset="mappings_src"),
            ),
        ],
        test_mode=False,
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        cache=False,
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
            # dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=1,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "segment", "instance"),
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "images",
                    "mappings_src",
                    "mappings_tgt",
                    'origin_coord', 'origin_segment', 'origin_instance',
                    "path",
                    "name",
                    "inverse"
                ),
                feat_keys=('coord'),
                # offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord", image_offset="images", mappings_offset="mappings_src"),
            ),
        ],
        test_mode=False,
        classes=classes,   
    ),
    
    test=dict(  
            type='MechanicalAssembly',
            split='train',
            data_root='data/scans',
            transform=[
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
                    keys=('coord', 'segment', 'instance', 'seg_indices')),
                dict(type='CenterShift', apply_z=False),
                dict(
                    type='InstanceParser',
                    segment_ignore_index=(-1, ),
                    instance_ignore_index=-1),
                dict(type='FPSSeed', n_points=100),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'segment', 'instance',
                        'origin_coord', 'origin_segment', 'origin_instance',
                        'instance_centroid', 'bbox', 'seed_ids', 'path',
                        'seg_indices'),
                    feat_keys='coord',
                    offset_keys_dict=dict(
                        offset='coord', origin_offset='origin_coord'))
            ],
            test_mode=False,
            classes={"other": 0, 
                        "gear": 1, 
                        "nut": 2, 
                        "screw": 3, 
                        "axe": 4})
)
