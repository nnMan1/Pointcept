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


class SuperpointPooling(nn.Module):

    def __init__(self, pool_function = torch_scatter.scatter_mean):
        super().__init__()

        self.pool_function = pool_function

    def __prepare_seg_indices(self, seg_indices, offset):

        bs = 0
        off = 0

        offs = []

        for be in offset:
            tmp = seg_indices[bs:be]
            _, inverse_indices = torch.unique(tmp, return_inverse=True)
            seg_indices[bs:be] = inverse_indices + off
            off = seg_indices[:be].max() + 1 
            offs.append(off)
            bs = be
    
        return seg_indices, torch.tensor(offs)

    def forward(self, data, keys=['instance', 'segment', 'features']):

         if 'seg_indices' not in data.keys():
               return data

         data['offset_orig'] = data['offset']
         data['seg_indices'], data['offset'] = self.__prepare_seg_indices(data['seg_indices'], data['offset'])

         label_keys = []
         if 'instance' in keys:
            label_keys.append('instance')
            keys.remove('instance')

         if 'segment' in keys:
            label_keys.append('segment')
            keys.remove('segment')

         
         for key in label_keys:
            label = []
            for cls in np.unique(data['seg_indices']):
               cluster_mask = data['seg_indices'] == cls
          
               unique_labels, counts = torch.unique(data[key][cluster_mask], return_counts=True)
               print(unique_labels, counts)
               majority_label = unique_labels[torch.argmax(counts)]
               label.append(majority_label)

            data[key] = np.asarray(label)

         for key in keys:
            data[key] = torch_scatter.scatter_mean(data[key],  data['seg_indices'], dim=0)
        
         return data


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
                                    grid_size=1.5,
                                    hash_type="fnv",
                                    mode="train",
                                    return_grid_coord=True,
                                    keys=("coord", "normal", "segment", "instance"),
                                ),
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
   pcd.colors = o3d.utility.Vector3dVector(colors[s['segment']])

   vis = o3d.visualization.Visualizer()
   vis.create_window(window_name=s['path'])
   vis.add_geometry(pcd)
   vis.run()
   vis.destroy_window()

   # o3d.visualization.draw_geometries([pc 
