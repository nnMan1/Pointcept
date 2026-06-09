import glob
import sys
import os
import json
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from segmentator import segment_mesh
import re

colors = np.random.randint(0, 255, (1000, 3)) / 255

data_root = 'data/abc_dataset/scans_smooth'
threshold = 0.01

assebmlies = glob.glob(f'{data_root}/*')

for i, assembly in enumerate(assebmlies):
    print(f"Processing assembly {i+1}/{len(assebmlies)}: {assembly}")

    if not os.path.isdir(assembly):
        continue

    if not os.path.exists(f'{assembly}/visible1.ply'):
        print(f"File {assembly}/visible1.ply does not exist, skipping...", file=sys.stderr)
        continue

    print("/////////////////")
    mesh = o3d.io.read_triangle_mesh(f'{assembly}/visible1.ply')

    if 'seg_indices' in json.load(open(f'{assembly}/1.json')).keys():
        continue
    
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()

    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.asarray(mesh.triangles, dtype=np.int64))
    ind1 = segment_mesh(vertices, faces, 0.0001, 5).numpy()
    ind2 = segment_mesh(vertices, faces, 0.00001, 5).numpy()

    frames = glob.glob(os.path.join(assembly, '*.ply'))
    frames = [p for p in frames if os.path.isfile(p) and re.fullmatch(r'^\d+\.ply$', p.split('/')[-1])]
    frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    labels = glob.glob(os.path.join(assembly, '*.json'))
    labels = [p for p in labels if os.path.isfile(p) and re.fullmatch(r'^\d+\.json$', p.split('/')[-1])]
    labels = sorted(labels, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    kd_tree = cKDTree(vertices.numpy())
    print(os.path.join(assembly, '[0-9]+.ply'))

    for f, l in zip(frames, labels):
        ann = json.load(open(l))
        pcd = o3d.io.read_point_cloud(f)
        
        points = np.asarray(pcd.points)
        _, indices = kd_tree.query(points, k=1)

        seg_indices1 = ind1[indices]
        seg_indices2 = ind2[indices]
        
        ann['seg_indices'] = seg_indices1.tolist()
        ann['seg_indices2'] = seg_indices2.tolist()

        with open(l, 'w') as outfile:
            json.dump(ann, outfile, indent=4)

    # o3d.visualization.draw_geometries([mesh], point_show_normal=True)