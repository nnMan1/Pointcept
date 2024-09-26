import sys
sys.path.append('/home')

import os
import torch
import numpy as np
import open3d as o3d 
from pointcept.datasets import build_dataset
from pointcept.utils.visualization import to_o3d, colors

dataset = build_dataset(dict(
                        type='Fuselage',
                        split='val_lr',
                        data_root='data/fuselage/crops_250x250x250_3_rivets',
                        transform=[
                            dict(type="CenterShift", apply_z=True),
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
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
                            dict(
                                type="GridSample",
                                grid_size=0.3,
                                hash_type="fnv",
                                mode="train",
                                keys=("coord", "segment", "normal", "seg_indices"),
                                return_grid_coord=True,
                            ),
                            dict(type="SphereCrop", point_max=100000, mode="random"),
                            dict(type="CenterShift", apply_z=False),
                            # dict(type="NormalizeColor"),
                            # dict(type="ShufflePoint"),
                            dict(type="ToTensor"),
                            dict(
                                type="Collect",
                                keys=("coord", "grid_coord", "segment", "seg_indices", 'path'),
                                feat_keys=("grid_coord"),
                            ),
                        ],
                        test_mode=False,
                        classes=[
                            'body', 'body1', 'panel', 'rivets_t1', 'rivets_t2', 'rivets_t3'
                        ]))

results = '/home'

for sample in dataset:
    # print(sample["id"])
    # result = np.load(os.path.join(results, f"{'_'.join(sample['path'].split('/'))}_pred.npy"))
    pcd = to_o3d(sample['coord'], verts_colors=colors[sample["seg_indices"] % len(colors)])

    o3d.io.write_point_cloud(os.path.join(results, f"{'_'.join(sample['path'].split('/'))}_pred.ply"), pcd)