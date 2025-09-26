_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8 # bs: total bs in all gpus
num_worker = 16
mix_prob = 0
empty_cache = True
enable_amp = False
evaluate = True
find_unused_parameters = True
# weight = 'exp/abc_dataset/insseg-myspformer_ptv3-v1m1-0-spunet-base/model/model_last.pth'
# resume = True# weight='backbones/sstnet_pretrain.pth'


num_classes = 1
fts_sizes = 128
dim_feedforward=1024
segment_ignore_index = (-1, )

# model settings
model = dict(
    type="MySPFormer",
    num_query = 100,
    encoder=dict(
        backbone=dict(
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
                dropout=0
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            )
        ],
    ),
    use_superpoint_pooling=False,
    instance_ignore_index=-1,
)



# scheduler settings
epoch = 500
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.002)
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
data_root = "data/segment-assembly-synthetic/data"
recompute_clustering=False

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
                    "images",
                    "mappings_src",
                    "mappings_tgt",
                    # "instance_centroid",
                    # "bbox",
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
        cache=True,
        recompute_clustering=recompute_clustering,
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
                keys=("coord", "segment", "instance", "seg_indices"),
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
                    "images",
                    "mappings_src",
                    "mappings_tgt",
                    'origin_coord', 'origin_segment', 'origin_instance',
                    # "instance_centroid",
                    # "bbox",
                    "seg_indices",
                    "path",
                    "name",
                    "inverse"
                ),
                feat_keys=('coord'),
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
    type="InsSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=False,
)