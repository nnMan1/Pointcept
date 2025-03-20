import os
import os.path as osp
import open3d as o3d
import json
import numpy as np
import trimesh

files = open('data/files.txt').readlines()

for file in files:
    file = file[2:].strip()
    dir = osp.dirname(file)

    print(file)

    with open(osp.join('data', dir, 'annotations.json')) as json_file:
        annotations = json.load(json_file)

    if not osp.exists(f'data/{file}'):
        print(f"File data/{file} does not exist.")
        continue
    
    mesh = trimesh.load(f'data/{file}')

    colors = np.array(np.random.randint(0, 256, (500, 3))[annotations['instance_id']])
    mesh.visual.vertex_colors = colors
    
    mesh.show()

