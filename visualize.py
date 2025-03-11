import numpy as np
import open3d as o3d
import glob

files = glob.glob('exp/fuselage_lr_split/semseg-pt-v3-0-base-ce-loss/result/P2*.ply')
print(files)

pcd = o3d.geometry.PointCloud()

for file in files:
    pcd += o3d.io.read_point_cloud(file)



o3d.visualization.draw_geometries([pcd])
