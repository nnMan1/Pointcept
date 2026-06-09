import os
import os.path as osp
import numpy as np
import trimesh
import glob
import json
from pointcept.datasets import build_dataset
from pointcept.datasets.transform import GridSample, Compose
from common_tools import VData
from pointcept.datasets.mcb_dataset import MCBDataset

from pointcept.models import build_model
import torch

class_names = [
    'Eye screws',
    'Setscrew',
    'Tapping screws',
    'Cap nuts',
    'Castle nuts',
    'Flange nut',
    'Hexagonal nuts',
    'Locknuts',
    'Rivet nut',
    'Slotted nuts',
    'Square nuts',
    'T-nut',
    'Wingnuts',
    'Screws and bolts with countersunk head',
    'Screws and bolts with cylindrical head',
    'Screws and bolts with hexagonal head',
    'Washer bolt',
    'other'
]

class_idx  = {
    'Eye screws': 0,
    'Setscrew': 0,
    'Tapping screws': 0,
    'Cap nuts': 1,
    'Castle nuts': 1,
    'Flange nut': 1,
    'Hexagonal nuts': 1,
    'Locknuts': 1,
    'Rivet nut': 1,
    'Slotted nuts': 1,
    'Square nuts': 1,
    'T-nut': 1,
    'Wingnuts': 1,
    'Screws and bolts with countersunk head': 0,
    'Screws and bolts with cylindrical head': 0,
    'Screws and bolts with hexagonal head': 0,
    'Washer bolt' :0,
    'other': 2
}

final_class_name = ['screw', 'nut', 'other']

for name in final_class_name:
    if not osp.exists(name):
        os.makedirs(name)

model_config = dict(
     type='DefaultClassifier',
    num_classes=3,
    backbone_embed_dim=256,
    backbone=dict(
        type='SpUNet-v1m1',
        in_channels=3,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode=True),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(type='FocalLoss', loss_weight=1.0, ignore_index=-1)
    ])

model = build_model(model_config)

checkpoint_path = 'exp/cls_part_dataset/cls-spunet-v1m1-0-base-v3/model/model_best.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

model = model.cuda().float()
model.eval()

# assemblies = glob.glob(osp.join('data/abc_dataset/chunks/*/stl3', '*'))
assemblies = []
for dirs in glob.glob('data/my_synth/raw/**/*.stl', recursive=True) + glob.glob('data/my_synth/raw/**/*.obj', recursive=True) + glob.glob('data/my_synth/raw/**/*.ply', recursive=True):
    print(dirs)
    dirname = osp.dirname(dirs)
    if dirname not in assemblies:
        assemblies.append(dirname)

print(f'Found {len(assemblies)} assemblies.')

# assemblies = [
#               'data/my_synth/raw/CouplingFalange/flange-coupling-15_struttura.STEP',
#               'data/my_synth/raw/CouplingFalange/flange-coupling-15.STEP',
#               'data/my_synth/raw/CouplingFalange/flange-coupling-21.STEP',
#               'data/my_synth/raw/Differential/stl3/2017-1-DIFFERENTIEL-ACHOU-BENASSON-BENRIDA-343-PARTS.stp',
#               'data/my_synth/raw/Differential/stl3/2017-1-DIFFERENTIEL-BUREL-BRATULIC-BENZAMIA-153-PARTS.stp',
#               'data/my_synth/raw/Differential/stl3/2017-1-DIFFERENTIEL-COSTE-ELDACHRI-CHAVIGNOT-209-PARTS.stp',
#               'data/my_synth/raw/Differential/stl3/2017-1-DIFFERENTIEL-LATETE-MALARD-MARTIN-246-PARTS.stp',
#               'data/my_synth/raw/Differential/stl3/2017-1-DIFFERENTIEL-LATHUILLE-GIODA-FRANCHETEAU-170-PARTS.stp',
#               'data/my_synth/raw/electromotors/1-1kw-1-5hp-4-pole-1400rpm-19mm-shaft-three-phase-electric-motor-reduced-80-frame-1.snapshot.1/stl4/1.1KW Reduced 80 Frame 3Ø Motor GL80-B3',
#               'data/my_synth/raw/electromotors/3PH-0.38 hp 1800RPM - MOTOR.STEP/*/',
#               'data/my_synth/raw/electromotors/12mt-bus-electric-motor-1.snapshot.2/stl3/12mt bus electric motor',
#             ]

transform = Compose(
        [
           dict(type='NormalizeCoord'),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='train',
                keys=('coord', ),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord'),
                feat_keys=['coord'])
        ])

num_classes = 25
rgb = np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)
alpha = np.full((num_classes, 1), 255, dtype=np.uint8)
colors = np.hstack([rgb, alpha])

c = 0

for i, assembly in enumerate(assemblies):
    parts = []
    labels = {}
    for part in glob.glob(osp.join(assembly, '*')):
        try:
            filename = osp.basename(part)
            part = trimesh.load_mesh(part)
            points = part.sample(4096, return_index=False) # weighted by face area by default

            # Align points with principal components
            # points_mean = points.mean(axis=0)
            # points_centered = points - points_mean
            # cov_matrix = np.cov(points_centered, rowvar=False)
            # eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
            # points_aligned = np.dot(points_centered, eig_vectors)
            # points = points_aligned + points_mean
            
            data = {'coord': points}
            data = transform(data)  
            # print(data['coord'].min(axis=0))
            # data['coord'] += data['coord'].min(axis=0)[0]

            for k, v in data.items():
                data[k] = v.cuda()    

            pred_label = model(data)['cls_logits'].argmax().item()
            # class_name = class_names[pred_label]
            # pred_label = class_idx[class_name]
            color = colors[pred_label]
            labels[filename] = final_class_name[pred_label]

            vertex_color = np.tile(color, (len(part.vertices), 1))

            # part.export(f'{final_class_name[pred_label]}/{c}.stl')
            # c += 1

            part.visual.vertex_colors = vertex_color
            
            
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(data['coord'].cpu().numpy())
            # o3d.visualization.draw_geometries([pcd])
            parts.append(part)
        except:
            continue
    #     part.export(f'{c}.ply')
    #     c += 1

    # exit(0)
    print(assembly)
    with open(f'{assembly}/meta.json', 'w') as f:
        json.dump(labels, f)

    # exit(0)

    # assembly_mesh = trimesh.util.concatenate(parts)
    # assembly_mesh.export(f'assembly{c}.ply')

    # c += 1
        
        

# dataset = MCBDataset(labels=['Chain drives'], include_other_classes=True)

# for b in dataset:
#     t = VData.from_dict({
#         'points': b['coord']
#     })

#     o3d.visualization.draw_geometries([t.to_o3d_pointcloud()])