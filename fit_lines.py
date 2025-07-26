import sys
sys.path.append('/home')

import os
import torch
import numpy as np
import open3d as o3d 
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.utils.visualization import to_o3d, colors
from sklearn.cluster import DBSCAN
from torch import nn
import torch_scatter
from sklearn.decomposition import PCA
from functools import partial
from torchvision import transforms
from pointcept.models.multivew.multiview_feaure_extraction import MeshFeatureExtractor


dataset = build_dataset(dict(
                        type = "MechanicalAssemblySynth",
                        data_root = "data/segment-assembly-synthetic/data",
                        recompute_clustering=False,
                        classes=dict({
                                    'other': 0,
                                    'screw': 1,
                                    'nut': 2,
                                }),                   
                                split='val',
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
                                    grid_size=5,
                                    hash_type="fnv",
                                    mode="train",
                                    return_inverse=True,
                                    return_grid_coord=True,
                                    keys=("coord", "normal", "segment", "instance", 'seg_indices'),
                                ),
                                # # dict(type="SphereCrop", point_max=1000000, mode='center'),
                                # dict(type="CenterShift", apply_z=False),
                                # dict(type="NormalizeColor"),
                                # dict(
                                #     type="InstanceParser",
                                #     segment_ignore_index=(-1, ),
                                #     instance_ignore_index=-1,
                                # ),
                                dict(type="ToTensor"),
                                dict(
                                    type="Collect",
                                    keys=(
                                        "coord",
                                        "grid_coord",
                                        "segment",
                                        "instance",
                                        "images",
                                        "mappings_src",
                                        "mappings_tgt",
                                        # "instance_centroid",
                                        # "bbox",
                                        "seg_indices",
                                        "path",
                                        "name",
                                        "inverse"
                                    ),
                                    feat_keys=("coord", "normal"),
                                    offset_keys_dict=dict(offset="coord", origin_offset="origin_coord", image_offset="images", mappings_offset="mappings_src"),
                                ),
                            ],test_mode=False,
                              image_transform=transforms.Compose([
                                        transforms.Resize((508, 508)),  # or 518 for ViT-Giant
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                    ])
                        ))

# dataset.update_semantic_labels('obj_assemblies')
# exit(0)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=2,
                                         collate_fn=partial(point_collate_fn),
                                        )

colors = np.random.randint(0, 255, (1500, 3)) / 255

feature_extractor = MeshFeatureExtractor(model_name="facebook/dinov2-small", device="cuda:0")

# dataset.prepare_clustering()

for s in dataloader:

    for key in s.keys():
        if isinstance(s[key], torch.Tensor):
            s[key] = s[key].cuda(non_blocking=True)
    
    with torch.no_grad():
        features = feature_extractor(s)

    bs = 0
    for i, be in enumerate(s['offset']):

        pca = PCA(n_components=9)
        tokens_pca = pca.fit_transform(features[bs:be].cpu())

        #    s = SuperpointPooling()(s, ['instance', 'segment'])
        #    s['segment'] = s['segment'][s['seg_indices']]

        # pcd = o3d.geometry.TriangleMesh()
        # pcd.vertices = o3d.utility.Vector3dVector(s['coord'])
        # pcd.triangles = o3d.utility.Vector3iVector(s['face'])
        # pcd.vertex_normals = o3d.utility.Vector3dVector(s['normal'])
        # pcd.vertex_colors = o3d.utility.Vector3dVector(colors[s['segment']])

        colors = tokens_pca[:, -3:]
        colors -= colors.min(axis=0)
        colors /= colors.max(axis=0)

    

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(s['coord'][bs:be].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors)

        #    vis = o3d.visualization.Visualizer()
        #    vis.create_window(window_name=s['path'])
        #    vis.add_geometry(pcd)
        #    vis.run()
        #    vis.destroy_window()


        # pcd.colors = o3d.utility.Vector3dVector(colors[s['instance'] % len(colors)])
        # print(s['coord'].shape, s['instance'].shape, s['segment'].shape, s['name'])
        print(f'{s["name"][i]}.ply')
        o3d.io.write_point_cloud(f'{s["name"][i]}.ply', pcd)
        bs = be
    exit(0)

    #    vis = o3d.visualization.Visualizer()
    #    vis.create_window(window_name=s['path'])
    #    vis.add_geometry(pcd)
    #    vis.run()
    #    vis.destroy_window()

    # o3d.visualization.draw_geometries([pc 
