import numpy as np
import open3d as o3d
import glob

files = glob.glob('exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_3_rivets/result/P3*.ply')
print(files)

pcd = o3d.geometry.PointCloud()

for file in files:
    pcd += o3d.io.read_point_cloud(file)



o3d.visualization.draw_geometries([pcd])
