import sys
sys.path.append('/home')

import os
import torch
import numpy as np
import open3d as o3d 
from pointcept.datasets import build_dataset
# from pointcept.utils.visualization import to_o3d, colors
from sklearn.cluster import DBSCAN

colors = np.random.randint(0, 255, (100, 3))

dataset = build_dataset(dict(
                        type='Fuselage',
                        split='train_lr',
                        data_root='data/Fuselage/crops_250x250x250_holes',
                        augment_holes=True,
                        transform=[],
                        test_mode=False,
                        classes=[
                           'body', 'body1', 'hole', 'panel', 'rivets'
                        ]))


colors = np.random.randint(0, 255, (20, 3)) / 255

for s in dataset:
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(s['coord'])
    pcd.colors = o3d.utility.Vector3dVector(colors[s['segment']])

    o3d.io.write_point_cloud('tmp.ply', pcd)
    exit(0)