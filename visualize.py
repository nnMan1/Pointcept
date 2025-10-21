import sys
sys.path.append('/home')

import os
import torch
import numpy as np
from common_tools import VData
import open3d as o3d 
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.visualization import to_o3d, colors
from sklearn.cluster import DBSCAN
from torch import nn
import torch_scatter
from sklearn.decomposition import PCA
from functools import partial
from torchvision import transforms
from pointcept.models.multivew.multiview_feaure_extraction import MeshFeatureExtractor


dataset = build_dataset(dict(
        type='MechanicalAssemblyV2',
        split='val',
        data_root='data/cetim_assembly/data',
        cache=True,
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
                grid_size=1,
                hash_type='fnv',
                mode='train',
                return_inverse=True,
                return_grid_coord=True,
                keys=('coord', 'segment', 'instance', 'seg_indices')),
            dict(type='CenterShift', apply_z=False),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance', 'images',
                      'mappings_src', 'mappings_tgt', 'origin_coord',
                      'origin_segment', 'origin_instance', 'seg_indices',
                      'path', 'name', 'inverse'),
                feat_keys='coord',
                offset_keys_dict=dict(
                    offset='coord',
                    origin_offset='origin_coord',
                    image_offset='images',
                    mappings_offset='mappings_src'))
        ],
        test_mode=False,
        classes=dict(other=0, gear=0, nut=0, screw=0, axe=0)))

model = build_model(dict(
    type='MySPFormer',
    num_query=100,
    encoder=dict(
        backbone=dict(
            input_channel=3,
            blocks=5,
            block_reps=2,
            media=32,
            normalize_before=True,
            return_blocks=True,
            pool='mean'),
        backbone_out_channels=32,
        out_channels=32),
    decoder=dict(
        in_channels=32,
        hlevels=6,
        mask_modules=[
            dict(
                num_classes=1, return_attn_masks=True, hidden_dim=128, reuse=1)
        ],
        query_refinement_modules=[
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0),
            dict(
                in_channels=128,
                mask_dim=128,
                dim_feedforward=1024,
                pre_norm=False,
                num_heads=8,
                dropout=0)
        ]),
    instance_ignore_index=-1))

model = model.cuda()
model.load_state_dict(torch.load('exp/abc_dataset/insseg-myspformer-v1m1-0-spunet-base_fix_mapping2/model/model_best.pth')['state_dict'])
model.eval()
print(model)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         collate_fn=partial(point_collate_fn),
                                        )

colors = np.random.randint(0, 255, (1500, 3)) / 255

feature_extractor = MeshFeatureExtractor(model_name="facebook/dinov2-small", device="cuda:0")

for i, s in enumerate(dataloader):
    print(i, ": ", s['name'])

    print(s['origin_coord'].shape)
    print(s['coord'].shape)

    with torch.no_grad():
        for key in s.keys():
            try:
                s[key] = s[key].cuda()
            except:
                pass    
        pred = model(s)

    sorted_indices = pred['pred_scores'].argsort()[::-1]
    pred['pred_scores'] = pred['pred_scores'][sorted_indices]
    pred['pred_masks'] = pred['pred_masks'][sorted_indices]

    pred['pred_masks'] = pred['pred_masks'] * pred['pred_scores'][..., None]

    mask = pred['pred_masks'].argmax(0)

    # print(pred.keys())
    # for score, mask in zip(pred['pred_scores'], pred['pred_masks']):
    #     if score < 0.2:
    #         break    
    #     print(score, mask)
    #     print(mask.sum())
    data = VData.from_dict({
        'points': s['coord'].cpu(),
        'labels': mask.astype(np.int32),
    })

    # o3d.visualization.draw_geometries([data.to_o3d_pointcloud(color='labels')], point_show_normal=False)

    o3d.io.write_point_cloud(f'data2_{i}.ply', data.to_o3d_pointcloud(color='labels'))

    # data = VData.from_dict({
    #     'points': b['coord'].cpu(),
    #     'border_dist': b['border_dist'].cpu() 
    # })

    # o3d.io.write_point_cloud(f'data2_gt_{i}.ply', data.to_o3d_pointcloud(color='border_dist'))
