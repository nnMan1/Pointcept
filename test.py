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

# dataset settings
dataset_type = "MechanicalAssemblySynth"
data_root = "data/abc_dataset/scans_smooth"

classes=dict({
            'other': 0,
            'nut': 1,
            'screw': 2
        })

class_names = ["other", "nut", "screw"]


dataset = build_dataset(dict(
                        type=dataset_type,
                        split='train',
                        data_root=data_root,
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
                                # dict(
                                #     type="GridSample",
                                #     grid_size=1.5,
                                #     hash_type="fnv",
                                #     mode="train",
                                #     return_grid_coord=True,
                                #     keys=("coord", "normal", "segment", "instance"),
                                # ),
                                # # dict(type="SphereCrop", point_max=1000000, mode='center'),
                                # dict(type="CenterShift", apply_z=False),
                                # dict(type="NormalizeColor"),
                                # dict(
                                #     type="InstanceParser",
                                #     segment_ignore_index=(-1, ),
                                #     instance_ignore_index=-1,
                                # ),
                                # dict(type="ToTensor"),
                                # dict(
                                #     type="Collect",
                                #     keys=(
                                #         "coord",
                                #         "grid_coord",
                                #         "segment",
                                #         "instance",
                                #         "origin_coord",
                                #         "origin_segment",
                                #         "origin_instance",
                                #         "instance_centroid",
                                #         "bbox",
                                #         "seg_indices",
                                #         "path"
                                #     ),
                                #     feat_keys=("coord", "normal"),
                                #     offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                                # ),
                            ],test_mode=False,
                        classes=classes))

colors = np.random.randint(0, 255, (1500, 3)) / 255

for i, s in enumerate(dataset):
   
   if i > len(dataset) - 1:
       break   
   
   i += 1

   print(s['path'])

   s['offset'] = [len(s['coord'])]
#    s = SuperpointPooling()(s, ['instance', 'segment'])
#    s['segment'] = s['segment'][s['seg_indices']]
    
   # pcd = o3d.geometry.TriangleMesh()
   # pcd.vertices = o3d.utility.Vector3dVector(s['coord'])
   # pcd.triangles = o3d.utility.Vector3iVector(s['face'])
   # pcd.vertex_normals = o3d.utility.Vector3dVector(s['normal'])
   # pcd.vertex_colors = o3d.utility.Vector3dVector(colors[s['segment']])
   pcd = o3d.geometry.PointCloud()
   pcd.points = o3d.utility.Vector3dVector(s['coord'])
   pcd.colors = o3d.utility.Vector3dVector(colors[s['seg_indices2'] % len(colors)])

   cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
   pcd = pcd.select_by_index(ind)

#    vis = o3d.visualization.Visualizer()
#    vis.create_window(window_name=s['path'])
#    vis.add_geometry(pcd)
#    vis.run()
#    vis.destroy_window()

   # o3d.visualization.draw_geometries([pc 
