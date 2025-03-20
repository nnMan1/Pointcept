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
                        transform=[],
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

colors = np.random.randint(0, 255, (200, 3)) / 255

for s in dataset:
    
   # pcd = o3d.geometry.TriangleMesh()
   # pcd.vertices = o3d.utility.Vector3dVector(s['coord'])
   # pcd.triangles = o3d.utility.Vector3iVector(s['face'])
   # pcd.vertex_normals = o3d.utility.Vector3dVector(s['normal'])
   # pcd.vertex_colors = o3d.utility.Vector3dVector(colors[s['segment']])
   pcd = o3d.geometry.PointCloud()
   pcd.points = o3d.utility.Vector3dVector(s['coord'])
   pcd.colors = o3d.utility.Vector3dVector(colors[s['segment']])

   vis = o3d.visualization.Visualizer()
   vis.create_window(window_name=s['path'])
   vis.add_geometry(pcd)
   vis.run()
   vis.destroy_window()

   # o3d.visualization.draw_geometries([pc 
