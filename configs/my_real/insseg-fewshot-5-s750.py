weight = '/home/exp/my_synth/insseg-motorft-rescale/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 59429542
save_path = 'exp/my_synth/insseg-motorft-rescale'
num_worker = 32
prefetch_factor = 2
batch_size = 8
batch_size_val = None
batch_size_test = None
epoch = 1200
eval_epoch = 15
sync_bn = True
enable_amp = True
empty_cache = True
find_unused_parameters = True
mix_prob = 0
param_dicts = None
hooks = [
    dict(
        type='CheckpointLoader', keywords=['module.'],
        replacement=['module.']),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(
        type='InsSegEvaluator',
        segment_ignore_index=(-1, ),
        instance_ignore_index=-1),
    dict(type='CheckpointSaver', save_freq=None)
]
train = dict(type='DefaultTrainer')
test = dict(
    type='InstSegTester',
    verbose=False,
    segment_ignore_index=(-1, ),
    instance_ignore_index=-1)
keep_on_cpu = ('image_features', )
features_root_synth = '/leonardo_scratch/large/userexternal/vdosljak/dino_features/motor_synth/features'
features_root_real = '/leonardo_scratch/large/userexternal/vdosljak/dino_features/cetim_real/features'
num_classes = 1
fts_sizes = 128
dim_feedforward = 1024
segment_ignore_index = (-1, )
instance_ignore_index = -1
border_radius = 4.0
model = dict(
    type='MySPFormer',
    num_query=100,
    encoder=dict(
        backbone=dict(
            type='MergeFeatures',
            model1_config=dict(
                type='Image2PointCLoud',
                model_type=None,
                model_name=None,
                fts_dim=1280,
                out_fts_dim=256,
                merge_strategy='random_sample',
                return_features=['feat'],
                freeze_backbone=False,
                freeze_backbone_bn=False,
                proj_norm=True),
            model2_config=dict(
                type='PT-V3FeatureExtractor',
                in_channels=3,
                order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
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
                pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D'),
                return_features=['feat'],
                fusion='concat',
                concat_norm=True)),
        backbone_out_channels=320,
        out_channels=32),
    decoder=dict(
        in_channels=32,
        hlevels=6,
        mask_modules=[
            dict(
                num_classes=1, return_attn_masks=True, hidden_dim=128, reuse=1)
        ],
        query_refinement_modules=[
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0.005),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0.005),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0.005),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0.005),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0.005),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0.005)
        ]),
    matcher=dict(
        type='HungarianMatcher',
        cost_terms=[
            dict(type='ClassCost', weight=0.5, enabled=True, use_logits=False),
            dict(
                type='MaskBCECost',
                weight=1.0,
                enabled=True,
                instance_ignore_index=-1),
            dict(
                type='MaskDiceCost',
                weight=1.0,
                enabled=True,
                instance_ignore_index=-1)
        ],
        instance_ignore_index=-1),
    use_superpoint_pooling=False,
    mask_dice_loss_weight=7.0,
    border_loss_weight=0.0,
    border_focal_alpha=0.5,
    mask_selection=dict(n_point_thr=50))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.0001,
    pct_start=0.01,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'MechanicalAssemblySynth'
data_root = 'data/segment-motor-synthetic/data'
recompute_clustering = False
image_size = (512, 512)
classes = dict(other=0, gear=0, nut=0, screw=0, axe=0)
class_names = ['other']
data = dict(
    num_classes=1,
    ignore_index=-1,
    names=['other'],
    train=dict(
        type='MechanicalAssemblyV2',
        split='fs5',
        image_size=(512, 512),
        data_root='/home/data/cetim_assembly/dataset/downsampled1',
        load_image_files=False,
        transform=[
            dict(
                type='LoadImageFeatures',
                features_root=
                '/leonardo_scratch/large/userexternal/vdosljak/dino_features/cetim_real/features'
            ),
            dict(type='CenterShift', apply_z=True),
            dict(type='RandomRescaleDiag', lo=80, hi=480),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomRotate', angle=[-1, 1], axis='x', p=0.5),
            dict(type='RandomRotate', angle=[-1, 1], axis='y', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.8),
            dict(type='RandomJitter', sigma=0.001, clip=0.02),
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
                keys=('coord', 'normal', 'segment', 'instance',
                      'seg_indices')),
            dict(type='CenterShift', apply_z=False),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='GenerateBoundary', radius=4.0),
            dict(type='RandomSeed', n_points=100),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'face', 'grid_coord', 'segment', 'instance',
                      'border', 'instance_segment', 'image_features',
                      'mappings_src', 'mappings_tgt', 'origin_coord',
                      'origin_segment', 'origin_instance', 'seed_ids',
                      'seg_indices', 'path', 'name', 'inverse'),
                feat_keys=('coord', ),
                offset_keys_dict=dict(
                    offset='coord',
                    origin_offset='origin_coord',
                    image_offset='image_features',
                    mappings_offset='mappings_src',
                    instance_segment_offset='instance_segment'))
        ],
        test_mode=False,
        classes=dict(other=0, gear=0, nut=0, screw=0, axe=0),
        loop=80),
    val=dict(
        type='MechanicalAssemblyV2',
        split='fs5',
        image_size=(512, 512),
        data_root='/home/data/cetim_assembly/dataset/downsampled1',
        load_image_files=False,
        transform=[
            dict(
                type='LoadImageFeatures',
                features_root=
                '/leonardo_scratch/large/userexternal/vdosljak/dino_features/cetim_real/features'
            ),
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
                keys=('coord', 'normal', 'segment', 'instance',
                      'seg_indices')),
            dict(type='CenterShift', apply_z=False),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='GenerateBoundary', radius=4.0),
            dict(type='FPSSeed', n_points=100),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'face', 'grid_coord', 'segment', 'instance',
                      'border', 'instance_segment', 'image_features',
                      'mappings_src', 'mappings_tgt', 'origin_coord',
                      'origin_segment', 'origin_instance', 'seed_ids',
                      'seg_indices', 'path', 'name', 'inverse'),
                feat_keys=('coord', ),
                offset_keys_dict=dict(
                    offset='coord',
                    origin_offset='origin_coord',
                    image_offset='image_features',
                    mappings_offset='mappings_src',
                    instance_segment_offset='instance_segment'))
        ],
        test_mode=False,
        classes=dict(other=0, gear=0, nut=0, screw=0, axe=0)),
    test=dict(
        type='MechanicalAssemblyV2',
        split='fs_test',
        image_size=(512, 512),
        data_root='/home/data/cetim_assembly/dataset/downsampled1',
        load_image_files=False,
        transform=[
            dict(
                type='LoadImageFeatures',
                features_root=
                '/leonardo_scratch/large/userexternal/vdosljak/dino_features/cetim_real/features'
            ),
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
                keys=('coord', 'normal', 'segment', 'instance',
                      'seg_indices')),
            dict(type='CenterShift', apply_z=False),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='GenerateBoundary', radius=4.0),
            dict(type='FPSSeed', n_points=100),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'face', 'grid_coord', 'segment', 'instance',
                      'border', 'instance_segment', 'image_features',
                      'mappings_src', 'mappings_tgt', 'origin_coord',
                      'origin_segment', 'origin_instance', 'seed_ids',
                      'seg_indices', 'path', 'name', 'inverse'),
                feat_keys=('coord', ),
                offset_keys_dict=dict(
                    offset='coord',
                    origin_offset='origin_coord',
                    image_offset='image_features',
                    mappings_offset='mappings_src',
                    instance_segment_offset='instance_segment'))
        ],
        test_mode=False,
        classes=dict(other=0, gear=0, nut=0, screw=0, axe=0)))
