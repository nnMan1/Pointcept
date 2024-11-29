import torch
from common_tools import VData
import open3d as o3d
from pointcept.models import build_model
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.utils.visualization import to_o3d, colors
from functools import partial


dataset = build_dataset(dict(
                    type='Assembly',
                        split="train",
                        data_root='data/assembly',
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
                            dict(type="ClipFeature", key="border_dist", min_value=0, max_value=2),
                            dict(
                                type="GridSample",
                                grid_size=0.2,
                                hash_type="fnv",
                                mode="train",
                                keys=("coord", "segment", "instance", "border_dist"),
                                return_grid_coord=True,
                            ),
                            dict(type="SphereCrop", point_max=200000, mode="random",
                                 keys=("coord", "grid_coord" "segment", "instance", "border_dist"),
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
                            ))

dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=5,
            collate_fn=partial(point_collate_fn, mix_prob=0),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

ptv3_cfg =  dict(
    type="EdgesDetector",
    backbone=dict(
        type="PT-v3m1",
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
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

model = build_model(ptv3_cfg)

print(model)
exit()

for d in dataloader:

    data = VData.from_dict({
        'points': d['coord'],
        'labels': d['instance'],
        'border_dist': d['border_dist'] / 4
    })

    print(data.border_dist.max(), data.border_dist.min())

    o3d.visualization.draw_geometries([data.to_o3d_pointcloud(color='border_dist')])
