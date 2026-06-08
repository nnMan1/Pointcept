_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4 # bs: total bs in all gpus
num_worker = 32
mix_prob = 0
empty_cache = True
enable_amp = False
evaluate = True
sync_bn = True
find_unused_parameters = True
weight = 'exp/abc_dataset/insseg-myspformer-ptv3-dinov3-no_superpoints-large-v1m1-0-spunet-base_sync_bn/model/model_best.pth'
# resume = True# weight='backbones/sstnet_pretrain.pth'


num_classes = 1
fts_sizes = 128
dim_feedforward=1024
instance_ignore_index = -1
segment_ignore_index = (-1, )

model = dict(
    type="MySPFormer",
    num_query = 100,
    seg_ce_loss_weight=0.5,
    mask_ce_loss_weight=1.0,
    mask_dice_loss_weight=1.0,
    score_loss_weight=0.5,
    fts_loss_weight=0.00,
    triplet_loss_weight=0.0,
    overlap_loss_weight=0.0,
    triplet_margin=0.0,
    encoder=dict(
        backbone=dict(
        type="MergeFeatures",
       model1_config=dict(
                type='Image2PointCLoud',
                model_type='DinoV3',
                model_name='/home/backbones/dinov3-vith16plus-pretrain-lvd1689m',
                fts_dim=1280,
                out_fts_dim=256,
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
)

# scheduler settings
epoch = 200
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
dataset_type = "MechanicalAssemblySynthV2"
data_root = "data/segment-motor-synthetic/data/dataset_v2"
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
        cache=False,
        data_root=data_root,
        image_size=image_size,
        transform=[
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
                keys=("coord", "normal", "segment", "instance"),
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
                    "seed_ids",
                    "instance_segment",
                    "path",
                    "name",
                    "inverse"
                ),
                feat_keys=("coord"),
                offset_keys_dict=dict(
                    offset="coord",
                    origin_offset="origin_coord",
                    image_offset="images",
                    mappings_offset="mappings_src",
                    instance_segment_offset="instance_segment"),
            ),
        ],
        test_mode=False,
        classes=classes,
    ),
    val=dict(
        type='MechanicalAssemblySynthV2',
        split='all',
        cache=False,
        data_root='/home/data/cetim_assembly/dataset/downsampled-vtk',
        image_size=image_size,
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
                return_inverse=True,
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
                keys=('coord', 'face', 'grid_coord', 'segment', 'instance',
                      'instance_segment', 'images', 'mappings_src',
                      'mappings_tgt', 'origin_coord', 'origin_segment',
                      'origin_instance', 'seg_indices', 'path', 'name',
                      'inverse', 'instance_segment'),
                feat_keys='coord',
                offset_keys_dict=dict(
                    offset='coord',
                    origin_offset='origin_coord',
                    image_offset='images',
                    mappings_offset='mappings_src',
                    instance_segment_offset='instance_segment'))
        ],
        test_mode=False,
        classes=dict(other=0, gear=0, nut=0, screw=0, axe=0)),
    test=dict(
        type='MechanicalAssemblySynthV2',
        split='all',
        cache=False,
        data_root='/home/data/cetim_assembly/dataset/downsampled-vtk',
        image_size=image_size,
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
                return_inverse=True,
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
                keys=('coord', 'face', 'grid_coord', 'segment', 'instance',
                      'instance_segment', 'images', 'mappings_src',
                      'mappings_tgt', 'origin_coord', 'origin_segment',
                      'origin_instance', 'seg_indices', 'path', 'name',
                      'inverse', 'instance_segment'),
                feat_keys='coord',
                offset_keys_dict=dict(
                    offset='coord',
                    origin_offset='origin_coord',
                    image_offset='images',
                    mappings_offset='mappings_src',
                    instance_segment_offset='instance_segment'))
        ],
        test_mode=False,
        classes=dict(other=0, gear=0, nut=0, screw=0, axe=0)))


hooks = [
    dict(type="CheckpointLoader", keywords=["module.", "module.encoder.backbone.feature_extractor1.model", "MySPFormer__query.weight"], 
                                 replacement=["module.", "module.encoder.backbone.feature_extractor1.model.model", "dummy"]),
    # dict(type="CheckpointLoader", keywords=["module.", "decoder.mask_modules.0.class_embed_head"], replacement=["module.", "dummy."]),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator",
         segment_ignore_index=segment_ignore_index,
         instance_ignore_index=instance_ignore_index,),
    dict(type="CheckpointSaver", save_freq=None),
]

# Tester
test = dict(
    type="InstSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=instance_ignore_index,
    verbose=False,
)