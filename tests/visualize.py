import glob
import open3d as o3d

files = glob.glob('exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_grouping/result/P5*.ply')[::]

ply = o3d.geometry.PointCloud()

for file in files:
    ply += o3d.io.read_point_cloud(file)

o3d.visualization.draw_geometries([ply])