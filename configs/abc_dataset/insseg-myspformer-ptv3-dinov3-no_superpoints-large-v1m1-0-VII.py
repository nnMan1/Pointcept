weight = '../model/model_last.pth'
resume = False
evaluate = True
test_only = False
seed = 14361261
save_path = '.'
num_worker = 32
prefetch_factor = 2
batch_size = 8
batch_size_val = None
batch_size_test = None
# Base peaks ~e24 and degrades; shorter budget keeps LR schedule tighter.
epoch = 60
eval_epoch = 30
sync_bn = False
enable_amp = False
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
num_classes = 1
fts_sizes = 128
dim_feedforward = 1024
segment_ignore_index = (-1, )
instance_ignore_index = -1
# Decoder dropout for regularization (was 0 in base — caused early peak ~e24).
decoder_dropout = 0.1
model = dict(
    type='MySPFormer',
    num_query=100,
    encoder=dict(
        backbone=dict(
            type='MergeFeatures',
            model1_config=dict(
                type='Image2PointCLoud',
                model_type='DinoV3',
                model_name='/home/backbones/dinov3-vith16plus-pretrain-lvd1689m',
                fts_dim=1280,
                out_fts_dim=256,
                merge_strategy='random_sample',
                return_features=['feat'],
                freeze_backbone=True,
                freeze_backbone_bn=True),
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
                return_features=['feat'])),
        backbone_out_channels=64,
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
                dropout=decoder_dropout),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=decoder_dropout),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=decoder_dropout),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=decoder_dropout),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=decoder_dropout),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=decoder_dropout)
        ]),
    matcher=dict(
        type='HungarianMatcher',
        cost_terms=[
            # ClassCost weight 0.5 -> 2.0 (best III group setting was III_a6
            # with 2.0); stronger classifier signal for the matcher.
            dict(type='ClassCost', weight=2.0, enabled=True, use_logits=False),
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
    use_superpoint_pooling=False)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.002)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.0001,
    pct_start=0.01,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'MechanicalAssemblySynth'
data_root = 'data/segment-assembly-merged-synthetic/data'
recompute_clustering = False
classes = dict(other=0, gear=0, nut=0, screw=0, axe=0)
class_names = ['other']
data = dict(
    num_classes=1,
    ignore_index=-1,
    names=['class_names'],
    train=dict(
        type='MechanicalAssemblySynth',
        split='train',
        cache=True,
        # Image2PointCLoud hardcodes //= 16, so all images must be 512x512
        image_size=(512, 512),
        data_root='data/segment-assembly-merged-synthetic/data',
        recompute_clustering=False,
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    segment='origin_segment',
                    instance='origin_instance')),
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
                type='GridSample',
                grid_size=1,
                hash_type='fnv',
                mode='train',
                return_inverse=True,
                return_grid_coord=True,
                keys=('coord', 'normal', 'segment', 'instance',
                      'seg_indices')),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='RandomSeed', n_points=100),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance', 'images',
                      'mappings_src', 'mappings_tgt', 'seed_ids',
                      'seg_indices', 'instance_segment', 'path', 'name',
                      'inverse'),
                feat_keys='coord',
                offset_keys_dict=dict(
                    offset='coord',
                    origin_offset='origin_coord',
                    image_offset='images',
                    mappings_offset='mappings_src',
                    instance_segment_offset='instance_segment'))
        ],
        test_mode=False,
        classes=dict(other=0, gear=0, nut=0, screw=0, axe=0),
        loop=5),
    # VAL = real subset (val_files.txt, 131 samples, artec/reductor-albi).
    # Faster sim2real selection signal; flip back to synthetic val once
    # this config is validated.
    val=dict(
        type='MechanicalAssemblyV2',
        split='val',
        image_size=(512, 512),
        data_root='/home/data/cetim_assembly/dataset/downsampled1',
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
        type='MechanicalAssemblyV2',
        split='all',
        image_size=(512, 512),
        data_root='/home/data/cetim_assembly/dataset/downsampled1',
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
