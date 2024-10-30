import sys
sys.path.append('/home')

import os
import torch
import numpy as np
import open3d as o3d 
from pointcept.datasets import build_dataset
from pointcept.utils.visualization import to_o3d, colors
from sklearn.cluster import DBSCAN

def IoU(ids1, ids2):
    n_points = np.maximum(ids1.max(), ids2.max()) + 1
    t1 = np.zeros((n_points, ))
    t2 = np.zeros((n_points, ))

    t1[ids1] = 1
    t2[ids2] = 1
    return (t1 * t2).sum() / (t1.sum() + t2.sum() - (t1 * t2).sum())

dataset = build_dataset(dict(
                        type='Fuselage',
                        split='val_lr',
                        data_root='data/fuselage/crops_250x250x250_holes',
                        transform=[],
                        test_mode=False,
                        classes=[
                           'body', 'body1', 'hole', 'panel', 'rivets'
                        ]))

dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            drop_last=False,
            persistent_workers=True,
        )

results = 'exp/fuselage_hole_detection/semseg-spunet-v1-m1-0-base_lr_split_holes_aug_v3/result/'

preds = []
gts = []
coords = []

print(len(dataset))

for i, sample in enumerate(dataset):

    if i > len(dataset):
        break

    result = np.load(os.path.join(results, f"{'_'.join(sample['path'].split('/'))}_pred.npy"))
    gt = sample['segment']
    points = sample['coord']

    preds.append(result)
    gts.append(gt)
    coords.append(points)

    # result = gt == result

    # pcd = to_o3d(sample['coord'], verts_colors=colors[result % len(colors)])

tp, tn, fp, fn = 0, 0, 0, 0

preds = np.concatenate(preds)[::3]
gts = np.concatenate(gts)[::3]
coords = np.concatenate(coords)[::3]

ids_pred = np.where(preds == 2)[0]
ids_gts = np.where(gts == 2)[0]

print(len(ids_pred), len(ids_gts))

coords1 = coords[ids_pred]
pcd = to_o3d(coords1)
clusters1 = DBSCAN(eps=4, min_samples=2).fit(coords1).labels_

print(clusters1.max())

coords2 = coords[ids_gts]
clusters2 = DBSCAN(eps=4, min_samples=2).fit(coords2).labels_

d1 = np.zeros((clusters1.max()+1,))
d2 = np.zeros((clusters2.max()+1,))
o = 0

for c1 in np.unique(clusters1):

    if np.sum(clusters1 == c1) < 20:
        o += 1
        continue

    for c2 in np.unique(clusters2):
        if IoU(ids_pred[clusters1 == c1], ids_gts[clusters2 == c2]) > 0.5:
            d1[c1] = 1
            d2[c2] = 1

print(d1.sum(), d2.sum(), (1 - d1).sum() - o, (1 - d2).sum())