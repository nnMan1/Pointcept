import os
import os.path as osp
import open3d as o3d
import json
import numpy as np
import trimesh

labels = ["other", "gear", "nut", "screw", "axe", "rivet", "sting-stif", "ruber-seal", "main_panel", "hole", "rivet_t1", "rrivet_t2"]
labels = {l: i for i, l in enumerate(labels)}

colors = np.asarray([
            [144, 238, 144],   # Light Green
            [70, 130, 180],    # Steel Blue
            [255, 140, 0],     # Dark Orange
            [255, 215, 0],     # Gold
            [0, 255, 255],     # Aqua
            [0, 191, 255],     # Deep Sky Blue
            [34, 139, 34],     # Forest Green
            [255, 69, 0],      # Orange Red
            [138, 43, 226],    # Blue Violet
            [173, 216, 230],   # Light Blue
            [128, 0, 128],     # Purple
            [0, 0, 128],       # Navy
            [128, 128, 128],   # Gray
            [0, 255, 0],       # Lime
            [0, 0, 255],       # Blue
            [255, 255, 0],     # Yellow
            [0, 255, 255],     # Cyan
            [255, 0, 255],     # Magenta
            [255, 182, 193],   # Light Pink
            [255, 99, 71],     # Tomato
            [255, 228, 181],   # Moccasin
            [255, 222, 173],   # Navajo White
            [255, 160, 122],   # Light Salmon
            [255, 127, 80],    # Coral
            [240, 230, 140],   # Khaki
            [230, 230, 250],   # Lavender
            [216, 191, 216],   # Thistle
            [221, 160, 221],   # Plum
            [238, 130, 238],   # Violet
        ])

scans = ['data/raw_scans/panel-1/airplane panel 1NoTable.obj',
         'data/raw_scans/panel-2/Scan1NoTable.stl',
         'data/raw_scans/panel-3/Scan1.stl',
         'data/raw_scans/panel-4/Scan1.stl',
         'data/raw_scans/panel-5/cetim_fuselage_data_P5_test_part3_Raw_with_stickerDots_orig.stl',
         'data/raw_scans/panel1/15/Scan 1.stl',
         'data/raw_scans/panel2/scan1/Scan 1.stl',
         'data/raw_scans/panel3/4-4_rivets_damaged_4_missing/scan2/Scan 1.stl',
         'data/raw_scans/panel4/1-all_rivets_present/scan1/Scan 1.stl',
         'data/raw_scans/panel5/4-two_rivets_missing/NoTable.stl']


for i, s in enumerate(scans):
    print(i, s)
    dir = osp.dirname(s)
    anntoatoon = json.load(open(osp.join(dir, 'annotations.json')))
    mesh = trimesh.load(s)

    cls_id = {c:i for i, c in enumerate(anntoatoon['classes'])}
    sem = np.asanyarray(anntoatoon['semantic_id'])
    for cls in anntoatoon['classes']:
        sem[sem == cls_id[cls]] = labels[cls]

    print(mesh.vertices.shape, sem.shape)

    mesh.visual.vertex_colors = colors[sem] 
    print(colors[sem].shape)
    mesh.export(osp.join(f'{i}_colored_mesh.ply'))
    # input()
    # o3d.visualization.draw_geometries([pcd])
    # print(np.asarray(pcd.points))

    # with open(osp.join(scan_dir, 'points.json')) as f:
    #     points = json.load(f)
    #     print(points)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points)
    #     o3d.visualization.draw_geometries([pcd])

    # with open(osp.join(scan_dir, 'mesh.json')) as f:
    #     mesh = json.load(f)
    #     print(mesh)
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(mesh['vertices'])
    #     mesh.triangles = o3d.utility.Vector3iVector(mesh['triangles'])
    #     o3d.visualization.draw_geometries([mesh])