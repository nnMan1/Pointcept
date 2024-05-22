import os
os.system("rm -r samples/*.npy")


import torch
import numpy as np
from pointcept.datasets import ABCDataset, Assembly, Cetim, ScanNetDataset
from pointcept.datasets import transform 
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from functools import partial
import pointcept.utils.comm as comm
from pointcept.models import build_model
from collections import OrderedDict
from pointcept.utils.visualization import nms
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
# from pointcept.utils.visualization import nms
num_classes = 1
segment_ignore_index = (-1, 0, 1)

model = build_model(dict(
     type='PG-v1m1',
    backbone=dict(
        type='SpUNet-v1m1',
        in_channels=3,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2)),
    backbone_out_channels=96,
    semantic_num_classes=1,
    semantic_ignore_index=-1,
    segment_ignore_index=(-1, ),
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=300,
    cluster_propose_points=100,
cluster_min_points=50))

checkpoint = torch.load('exp/assembly/insseg-pointgroup-v1m1-0-spunet-base/model/model_last.pth')

weight = OrderedDict()

for key, value in checkpoint["state_dict"].items():
    if key.startswith("module."):
        if comm.get_world_size() == 1:
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
    else:
        if comm.get_world_size() > 1:
            key = "module." + key  # xxx.xxx -> module.xxx.xxx
    weight[key] = value

model.load_state_dict(weight)
model = model.cuda()
model.eval()

ds = ABCDataset(
        split='val',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    segment='origin_segment',
                    instance='origin_instance')),
            dict(
                type='GridSample',
                grid_size=0.5,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                keys=('coord', 'segment', 'instance')),
            dict(type='CenterShift', apply_z=False),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='FPSSeed', n_points=25),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance',
                      'origin_coord', 'origin_segment', 'origin_instance',
                      'instance_centroid', 'bbox', 'seed_ids'),
                feat_keys='coord',
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False)

dataloader = torch.utils.data.DataLoader(ds,
                                         batch_size=1,
                                        collate_fn=partial(point_collate_fn),
                                        )
for id, b in enumerate(dataloader):
    with torch.no_grad():
        for key in b.keys():
            try:
                b[key] = b[key].cuda()
            except:
                pass
            
        pred = model(b)

    
    coords = b['coord'].cpu().numpy()
    preds = pred['pred_masks'].T.cpu().numpy()
    gt = b['instance'].unsqueeze(-1).cpu().numpy()

    save = np.concatenate([coords, preds, gt], -1)
    np.save(f'samples/{id}.npy', save)
    
  
    