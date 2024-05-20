import os
os.system("rm -r samples/*.npy")


import torch
import numpy as np
from pointcept.datasets import ABCDataset, Assembly, Cetim
from pointcept.datasets import transform 
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from functools import partial
import pointcept.utils.comm as comm
from pointcept.models import build_model
from collections import OrderedDict

from pointcept.utils.visualization import nms

model = build_model(dict(
    type='Mask-3D',
    backbone=dict(
        type='MinkUNet34C', in_channels=3, out_channels=128, out_fpn=True),
    position_encoding=dict(
        type='PositionEmbeddingCoordsSine',
        pos_type='fourier',
        d_pos=128,
        gauss_scale=1,
        normalize=True),
    mask_module_config=dict(
        num_classes=1, return_attn_masks=True, use_seg_masks=False,         
        num_masks=3
        ),
    query_refinement_config=dict(pre_norm=False, num_heads=8, dropout=0),
    num_decoders=1,
    dim_feedforward=1024,
    hidden_dim=128,
    mask_dim=128))

checkpoint = torch.load('/home/exp/abc_dataset_my_matcher_multimask/insseg-mask3d-v1m1-0-spunet-base_3_masks_dice+focal/model/model_last.pth')

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
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.5),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.5),
            dict(type='NormalizeCoord'),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.8),
            dict(type='RandomJitter', sigma=0.001, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                keys=('coord', 'segment', 'instance')),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='FPSSeed', n_points=100),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance',
                      'instance_centroid', 'bbox', 'seed_ids', 'id', 'path', 'weights'),
                feat_keys='grid_coord')
        ],
        test_mode=False
)

dataloader = torch.utils.data.DataLoader(ds,
                                         batch_size=1,
                                        collate_fn=partial(point_collate_fn),
                                        )

for b in dataloader:
    
    with torch.no_grad():
        for key in b.keys():
            try:
                b[key] = b[key].cuda()
            except:
                pass
            
        pred = model(b)

    # if b['id'] != 22:
    #     continue;
    
    scores = pred['pred_score'][0].cpu().numpy()
    ious = pred['ious'][0].cpu().numpy()
    coords = b['coord'].cpu().numpy()
    preds = pred['masks'].cpu().numpy()#.argmax(-1, keepdims=True)

    preds1 = 1 / (1 + np.exp(-preds))
    stability = (preds1 > 0.8).sum(0) / ((preds1 > 0.2).sum(0) + 1e-5)

    # for iu, st, sc in zip(ious, stability, scores):
    #     print(iu, st, sc)
    
    filters = stability > 0.6
    preds = preds[:, filters]
    ious = ious[filters]
    scores = scores[filters]
    preds1 = preds1[:, filters]
    stability = stability[filters]

    # #remove compleate masks
    # full = (preds1 > 0.5).mean(0)
    # preds = preds[:, full < 0.8]
    # ious = ious[full < 0.8]
    # scores = scores[full < 0.8]
    # stability = stability[full < 0.8]

    # gt = np.expand_dims(pred['matched_targets'][0].cpu().numpy(), -1)
    gt = b['instance'].unsqueeze(-1).cpu().numpy()
    # preds = preds[:, ious > 0.3]
    # scores = ious[ious > 0.3]
    
    print(1, len(scores), len(preds.T))
    
    keep = nms(preds, scores, 0.3)
    preds = preds[:, keep]
    scores = scores[keep]
    ious = ious[keep]
  
    print(ious)
    print(preds.shape, coords.shape, gt.shape)
    
    save = np.concatenate([coords, preds, gt], -1)
    
    np.save(f'samples/{str(b["id"].cpu().numpy())}.npy', save)
