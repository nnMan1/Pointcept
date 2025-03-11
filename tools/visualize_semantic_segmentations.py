import sys
sys.path.append('/home')

import os
import torch
import numpy as np
import open3d as o3d 
from pointcept.datasets import build_dataset
from pointcept.utils.visualization import to_o3d, colors

dataset = build_dataset(dict(
                        type='Fuselage',
                        split='val',
                        data_root='data/Fuselage/new_data_prepared',
                        transform=[],
                        test_mode=False,
                        classes=[ 'sting-stif', 'other', 'main_panel', 'rivet', 'hole', 'ruber-seal', 'rivet_t1'],
                        merged_classes=[['rivet', 'rivet_t1'], ['sting-stif', 'ruber-seal']]
                        ))

dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            drop_last=False,
            persistent_workers=True,
        )

results = 'exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_holes_aug_v6/result'

for sample in dataset:
    result = sample['path']
    result = result.replace('/', '_')+'_pred.npy'
    result = np.load(f'{results}/{result}')
    gt = sample['segment']

    # result = gt == result

    pcd = to_o3d(sample['coord'], verts_colors=colors[result % len(colors)])

    o3d.io.write_point_cloud(os.path.join(results, f"{'_'.join(sample['path'].split('/'))}_pred.ply"), pcd)

