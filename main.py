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


dataset_type = "MechanicalAssembly"
data_root = "data/scans"

classes={"other": 0, 
         "gear": 0, 
         "nut": 0, 
         "screw": 0, 
         "axe": 0, }

class_names = ["other", "gear", "nut", "screw", "axe"]

dataset = build_dataset(dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        ignore_index=-1,
        classes=classes,
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
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1,
            ),
            dict(type='FPSSeed', n_points = 100),
            dict(type="ToTensor"),
            # dict(type="VoxelizeSuperpoints", voxel_size=5),
            dict(type="SuperpointPool", n_points=100),
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
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False))

colors = np.random.randint(0, 255, (1500, 3)) / 255

for s in dataset:

   s['offset'] = [len(s['coord'])]
   print(s['offset'])
   print(s['seg_indices'].max())
   print(s['coord'][s['superpoint_pooling']].shape)
#    s = SuperpointPooling()(s, ['instance', 'segment'])
#    s['segment'] = s['segment'][s['seg_indices']]
    
   # pcd = o3d.geometry.TriangleMesh()
   # pcd.vertices = o3d.utility.Vector3dVector(s['coord'])
   # pcd.triangles = o3d.utility.Vector3iVector(s['face'])
   # pcd.vertex_normals = o3d.utility.Vector3dVector(s['normal'])
   # pcd.vertex_colors = o3d.utility.Vector3dVector(colors[s['segment']])
   pcd = o3d.geometry.PointCloud()
   pcd.points = o3d.utility.Vector3dVector(s['coord'][s['superpoint_pooling']])
   pcd.colors = o3d.utility.Vector3dVector(colors[s['seg_indices'][s['superpoint_pooling']] % len(colors)])

   vis = o3d.visualization.Visualizer()
   vis.create_window(window_name=s['path'])
   vis.add_geometry(pcd)
   vis.run()
   vis.destroy_window()

   # o3d.visualization.draw_geometries([pc 