import sys
sys.path.append('/home')

import os
import torch
import numpy as np
import open3d as o3d 
from pointcept.datasets import build_dataset
from pointcept.utils.visualization import to_o3d, colors
from sklearn.cluster import DBSCAN
from torch import nn
import torch_scatter
from pointcept.models.utils.nn import SuperpointPooling, SuperpointUnpooling

dataset = build_dataset(dict(
                        type='MechanicalAssembly',
                        split='train',
                        data_root='data/crops',
                        augment_holes=True,
                        transform=[
                                dict(type="CenterShift", apply_z=True),
                                dict(
                                    type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
                                ),
                                # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
                                dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                                dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.5),
                                dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.5),
                                dict(type="RandomScale", scale=[0.9, 1.1]),
                                # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                                dict(type="RandomFlip", p=0.5),
                                dict(type="RandomJitter", sigma=0.005, clip=0.02),
                                # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                                dict(
                                    type="GridSample",
                                    grid_size=0.3,
                                    hash_type="fnv",
                                    mode="train",
                                    return_grid_coord=True,
                                    keys=("coord", "normal", "segment", "instance"),
                                ),
                                # dict(type="SphereCrop", sample_rate=0.8, mode="random"),
                                dict(type="CenterShift", apply_z=False),
                                dict(type="ToTensor"),
                                dict(
                                    type="Collect",
                                    keys=(
                                        "coord",
                                        "grid_coord",
                                        "segment",
                                        "instance"
                                    ),
                                    feat_keys=("coord", "normal"),
                                ),
                            ],
                            classes=dict({
                                        'other': 5,
                                        'gear': 5,
                                        'nut': 5,
                                        'screw': 1,
                                        'axe': 1,
                                        'rivet': 4,
                                        'sting-stif': 0,
                                        'ruber-seal': 1,
                                        'main_panel': 3,
                                        'hole': 2,
                                        'rivet_t1': 4,
                                        'rrivet_t2': 4
                                    })))

colors = np.random.randint(0, 255, (55500, 3)) / 255

for i, s in enumerate(dataset):

    # print(i)
    # if True:
    #     continue

    # s['offset'] = [len(s['coord'])]
    # s = SuperpointPooling()(s, ['instance', 'segment'])
    # s = SuperpointUnpooling()(s, ['instance', 'segment'])
    
#    pcd = o3d.geometry.TriangleMesh()
#    pcd.vertices = o3d.utility.Vector3dVector(s['coord'])
#    pcd.triangles = o3d.utility.Vector3iVector(s['face'])
#    pcd.vertex_normals = o3d.utility.Vector3dVector(s['normal'])
#    pcd.vertex_colors = o3d.utility.Vector3dVector(colors[s['segment']])
    print(s['coord'].shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(s['coord'])
    pcd.colors = o3d.utility.Vector3dVector(colors[s['segment']+1])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="test")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

   # o3d.visualization.draw_geometries([pc 
