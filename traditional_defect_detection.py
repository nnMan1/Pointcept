import torch
import numpy as np
from common_tools import VData
import open3d as o3d
from pointcept.models import build_model
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.utils.visualization import to_o3d, colors
from functools import partial
import pointcept.utils.comm as comm
from collections import OrderedDict
from common_tools import VData

dataset_type = "MechanicalAssembly"
data_root = "data/scans"

classes={"other": 0, 
         "gear": 0, 
         "nut": 0, 
         "screw": 0, 
         "axe": 0, }

class_names = ["other", "gear", "nut", "screw", "axe"]

dataset = build_dataset(dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        ignore_index=-1,
        classes=classes,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            # dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment", "instance", "seg_indices"),
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="InstanceParser",
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1,
            ),
            dict(type='FPSSeed', n_points = 100),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "seed_ids",
                    "path",
                    "seg_indices",
                ),
                feat_keys=('coord'),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False))


dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         collate_fn=partial(point_collate_fn),
                                        )

model = build_model(dict(
    type='SPFormer',
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

model.load_state_dict(torch.load('exp/abc_dataset/insseg-spformer-v1m1-0-spunet-base_mix_group/model/model_best.pth')['state_dict'])
model = model.cuda()
model.eval()

for i, b in enumerate(dataset):

    b['segment'][:] = 0

    # if i % 50 != 0:
    #     continue

    with torch.no_grad():
        for key in b.keys():
            try:
                b[key] = b[key].cuda()
            except:
                pass
            
        pred = model(b)

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
        'points': b['coord'].cpu(),
        'labels': mask.astype(np.int32),
    })

    # o3d.visualization.draw_geometries([data.to_o3d_pointcloud(color='labels')], point_show_normal=False)

    o3d.io.write_point_cloud(f'data2_{i}.ply', data.to_o3d_pointcloud(color='labels'))

    # data = VData.from_dict({
    #     'points': b['coord'].cpu(),
    #     'border_dist': b['border_dist'].cpu() 
    # })

    # o3d.io.write_point_cloud(f'data2_gt_{i}.ply', data.to_o3d_pointcloud(color='border_dist'))