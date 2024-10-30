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
                        split='val_lr',
                        data_root='data/fuselage/crops_250x250x250',
                        transform=[],
                        test_mode=False,
                        classes=[
                           'body', 'body1', 'panel', 'rivets'
                        ]))

dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            drop_last=False,
            persistent_workers=True,
        )

results = 'exp/fuselage_lr_split/semseg-pt-v3-0-base-ce-loss/result'

for sample in dataset:
    print(sample.keys())
    result = np.load(os.path.join(results, f"{'_'.join(sample['path'].split('/'))}_pred.npy"))
    gt = sample['segment']

    # result = gt == result

    pcd = to_o3d(sample['coord'], verts_colors=colors[result % len(colors)])

    o3d.io.write_point_cloud(os.path.join(results, f"{'_'.join(sample['path'].split('/'))}_pred.ply"), pcd)
