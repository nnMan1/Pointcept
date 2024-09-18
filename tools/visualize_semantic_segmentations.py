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
                        data_root='data/fuselage/crops_250x250x250_3_rivets',
                        transform=[],
                        test_mode=False,
                        classes=[
                            'body', 'body1', 'panel', 'rivets_t1', 'rivets_t2', 'rivets_t3'
                        ]))

dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            drop_last=False,
            persistent_workers=True,
        )

results = 'exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_3_rivets/result'

for sample in dataset:
    print(sample["id"])
    result = np.load(os.path.join(results, f"{'_'.join(sample['path'].split('/'))}_pred.npy"))
    pcd = to_o3d(sample['coord'], verts_colors=colors[result % len(colors)])

    o3d.io.write_point_cloud(os.path.join(results, f"{'_'.join(sample['path'].split('/'))}_pred.ply"), pcd)