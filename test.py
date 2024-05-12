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

# from pointcept.utils.visualization import nms

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
        ),
    query_refinement_config=dict(pre_norm=False, num_heads=8, dropout=0),
    num_decoders=1,
    dim_feedforward=1024,
    hidden_dim=128,
    mask_dim=128))

checkpoint = torch.load('exp/abc_dataset_hungarian_matcher/insseg-mask3d-v1m1-0-spunet-base_dice_loss+focall_loss/model/model_last.pth')

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
                      'instance_centroid', 'bbox', 'seed_ids', 'id', 'path'),
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

    coords = b['coord'].cpu().numpy()

    pred_ious = pred['pred_iou'][0].cpu().numpy()
    scores = pred['pred_score'][0].cpu().numpy()
    stability = pred['stability_score'][0].cpu().numpy()
    ious = pred['bious'][0].cpu().numpy()
    # pred_ious = pred['bious'][0].cpu().numpy() 
    preds = pred['masks'].cpu().numpy()

    # pred_ious = 1 / (1 + np.exp(-pred_ious))

    for sc, st, iu, piou in zip(scores, stability, ious, pred_ious):
        print(sc, st, iu, piou)

    filter = (stability > 0.6) #& (pred_ious > 0.3)

    preds = preds[:, filter]
    scores = scores[filter]
    ious = ious[filter]
    stability = stability[filter]
    pred_ious = pred_ious[filter]

    keep = nms(preds, stability, 0.3)
    preds = preds[:, keep]
    scores = scores[keep]
    
    # preds = pred['matched_masks'][0].cpu().numpy()
    gt = pred['matched_targets'][0].unsqueeze(-1).cpu().numpy()
  
    save = np.concatenate([coords, preds, gt], -1)
    
    np.save(f'samples/{str(b["id"].cpu().numpy())}.npy', save)