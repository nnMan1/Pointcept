import sys
sys.path.append('/home')

import os
import torch
import numpy as np
import open3d as o3d 
from pointcept.datasets import build_dataset
from pointcept.utils.visualization import to_o3d, colors
from sklearn.cluster import DBSCAN

dataset = build_dataset(dict(
                        type='MechanicalAssembly',
                        split='train',
                        data_root='data',
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
                              dict(
                                 type="GridSample",
                                 grid_size=1,
                                 hash_type="fnv",
                                 mode="train",
                                 return_grid_coord=True,
                                 keys=("coord", "normal", "segment", "instance", "seg_indices"),
                              ),
                              # dict(type="SphereCrop", point_max=1000000, mode='center'),
                              dict(type="CenterShift", apply_z=False),
                              dict(type="NormalizeColor"),
                              dict(
                                 type="InstanceParser",
                                 segment_ignore_index=(-1, ),
                                 instance_ignore_index=-1,
                              ),
                              dict(type="ToTensor"),
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
                                    "seg_indices"
                                 ),
                                 feat_keys=("coord", "normal"),
                                 offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                              ),
                        ],
                        test_mode=False,
                        classes={"other": 0, 
                                 "gear": -1, 
                                 "nut": -1, 
                                 "screw": -1, 
                                 "axe": -1, 
                                 "rivet": 1, 
                                 "sting-stif": 2, 
                                 "ruber-seal": 3, 
                                 "main_panel": 4,
                                 "hole": 5,
                                 "rivet_t1": 1,
                                 "rrivet_t2": 1}))

colors = np.random.randint(0, 255, (1500, 3)) / 255

for s in dataset:
    
   # pcd = o3d.geometry.TriangleMesh()
   # pcd.vertices = o3d.utility.Vector3dVector(s['coord'])
   # pcd.triangles = o3d.utility.Vector3iVector(s['face'])
   # pcd.vertex_normals = o3d.utility.Vector3dVector(s['normal'])
   # pcd.vertex_colors = o3d.utility.Vector3dVector(colors[s['segment']])
   pcd = o3d.geometry.PointCloud()
   pcd.points = o3d.utility.Vector3dVector(s['coord'])
   pcd.colors = o3d.utility.Vector3dVector(colors[s['seg_indices']])

   vis = o3d.visualization.Visualizer()
   vis.create_window(window_name="test")
   vis.add_geometry(pcd)
   vis.run()
   vis.destroy_window()

   # o3d.visualization.draw_geometries([pc 