_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4 # bs: total bs in all gpus
num_worker = 30
mix_prob = 0
empty_cache = True
enable_amp = False
evaluate = True
# find_unused_parameters = False
# resume=False
# weight='exp/abc_dataset/insseg-spformer-v1m1-0-spunet-base/model/model_last.pth'
# weight='backbones/sstnet_pretrain.pth'


num_classes = 1
fts_sizes = 128
dim_feedforward=1024
segment_ignore_index = (-1, )

# model settings
model = dict(
    type="SPFormer",
    num_query = 100,
    encoder=dict(
        backbone=dict(
            input_channel=3,
            blocks=5,
            block_reps=2,
            media=32,
            normalize_before=True,
            return_blocks=True,
            pool='mean'
        ),
        backbone_out_channels=32,
        out_channels=32,
     ),
     positional_embedding=dict(
        type='PositionEmbeddingCoordsSine',
        pos_type="fourier",
        d_pos=128,
        gauss_scale=1,
        normalize=True,
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
data_root = "data/abc_dataset/scans_smooth"

classes=dict({
            'other': 0,
            'nut': 0,
            'screw': 0
        })

class_names = ["other"]


data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=['class_names'],
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
                return_grid_coord=True,
                keys=("coord", "segment", "instance", "seg_indices"),
            ),
            dict(type="SphereCrop",  point_max=200000, mode="random"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(type="SuperpointPool", n_points=100),
            dict(type="SuperpointPool", n_points=5, index_key="instance", pool_key="seed_ids"),
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
                    "path",
                    "seg_indices",
                    "superpoint_pooling"
                ),
                feat_keys=("grid_coord"),
                offset_keys_dict=dict(offset="coord", seed_ids_offset="seed_ids", superpoint_pooling_offset="superpoint_pooling"),
            ),
        ],
        test_mode=False,
        classes=classes,
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
            # dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=1,
                hash_type="fnv",
                mode="train",
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
            dict(type="ToTensor"),
            dict(type="SuperpointPool", n_points=100),
            dict(type="SuperpointPool", n_points=1, index_key="instance", pool_key="seed_ids"),
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
                    "seed_ids",
                    "path",
                    "seg_indices",
                    "superpoint_pooling"
                ),
                feat_keys=('coord'),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord", seed_ids_offset="seed_ids", superpoint_pooling_offset="superpoint_pooling"),
            ),
        ],
        test_mode=False,
        classes=classes,   
    ),
    test=dict(),  # currently not available
)

hooks = [
    dict(type="CheckpointLoader", keywords=["module.", "module.encoder.backbone.input_conv.0"], replacement=["module.encoder.backbone.", "dummy"]),
    # dict(type="CheckpointLoader", keywords=["module.", "decoder.mask_modules.0.class_embed_head"], replacement=["module.", "dummy."]),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator",),
    dict(type="CheckpointSaver", save_freq=None),
]
