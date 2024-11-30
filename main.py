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
                            dict(type="ScaleValues", key="border_dist", min_value=2, max_value=0),
                            dict(
                                type="GridSample",
                                grid_size=0.2,
                                hash_type="fnv",
                                mode="train",
                                keys=("coord", "segment", "instance", "border_dist"),
                                return_grid_coord=True,
                            ),
                            dict(type="SphereCrop", point_max=200000, mode="random",
                                 keys=("coord", "grid_coord", "segment", "instance", "border_dist"),
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
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=1,
        channels=(32, 64, 128, 128, 96, 96),
        layers=(2, 3, 4, 2, 2, 2),
    ),
    criteria=[
        dict(type="BCELoss", loss_weight=1.0)
    ],
)

model = build_model(ptv3_cfg).cuda()

print(model)

for d in dataloader:

    for k in d.keys():
        try:
            d[k] = d[k].cuda()
        except:
            print(k)

    d['border_dist'] = d['border_dist'].float()
    # pred = model(d)
    # print(pred.keys())

    data = VData.from_dict({
        'points': d['coord'].cpu(),
        'labels': d['instance'].cpu(),
        'border_dist': d['border_dist'].cpu() / 4
    })

    print(data.border_dist.max(), data.border_dist.min())

    o3d.visualization.draw_geometries([data.to_o3d_pointcloud(color='border_dist')])
