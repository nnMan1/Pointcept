import torch
from common_tools import VData
import open3d as o3d
from pointcept.models import build_model
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.utils.visualization import to_o3d, colors
from functools import partial
import pointcept.utils.comm as comm
from collections import OrderedDict
from common_tools import VData

dataset_type = "ScanNetDataset"
data_root = "data/scannet_instance_seg"


class_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]
num_classes = 20
segment_ignore_index = (-1, 0, 1)

dataset = build_dataset(dict(
                    type=dataset_type,
                    split="val",
                    data_root=data_root,
                    transform=[
                        dict(type="CenterShift", apply_z=True),
                        # dict(type="NormalizeCoord"),
                        dict(
                            type="Copy",
                            keys_dict={
                                "coord": "origin_coord",
                                "segment": "origin_segment",
                                "instance": "origin_instance",
                            },
                        ),
                        dict(
                            type="GridSample",
                            grid_size=0.02,
                            hash_type="fnv",
                            mode="train",
                            return_grid_coord=True,
                            keys=("coord", "color", "normal", "segment", "instance", "seg_indices"),
                        ),
                        # dict(type="SphereCrop", point_max=1000000, mode='center'),
                        dict(type="CenterShift", apply_z=False),
                        dict(type="NormalizeColor"),
                        dict(
                            type="InstanceParser",
                            segment_ignore_index=segment_ignore_index,
                            instance_ignore_index=-1,
                        ),
                        dict(type="FPSSeed", n_points=100),
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
                                "seg_indices",
                                "group_segment"
                            ),
                            feat_keys=("color", "normal"),
                            offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                        ),
                    ],))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         collate_fn=partial(point_collate_fn),
                                        )

fts_sizes = 128
dim_feedforward=1024

model = build_model(dict(
     type="Mask-3D",
     encoder=dict(
        backbone=dict(
                type="Res16UNet34C",
                in_channels = 6,
                out_channels = 128,
                out_fpn=True, #return intermidiate features
            ),
        out_channels=fts_sizes
     ),
     decoder=dict(
        in_channels=fts_sizes,
        hlevels=5,
        positional_encoding=dict(
            type='PositionEmbeddingCoordsSine',
            pos_type="fourier",
            d_pos=128,
            gauss_scale=1,
            normalize=True,
        ),
        mask_modules=[
            dict(
                num_classes=num_classes, 
                return_attn_masks=True, 
                hidden_dim=fts_sizes,
                reuse=3
            )
        ],
        query_refinement_modules=[
            dict(
                in_channels=256,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                sample_size=200,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            ),
            dict(
                in_channels=256,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                sample_size=800,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            ),
            dict(
                in_channels=128,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                sample_size=3200,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            ),
            dict(
                in_channels=96,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                sample_size=12800,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            ),
            dict(
                in_channels=96,
                mask_dim=fts_sizes,
                dim_feedforward=dim_feedforward,
                sample_size=51200,
                pre_norm=False,
                num_heads=8, 
                dropout=0
            )
        ],
    ),
    instance_ignore_index=-1,
    ))

    # num_decoders=1,
    # hidden_dim=128,
    # mask_dim=128,

# checkpoint = torch.load('exp/scannet/insseg-mask3d-v1m1-0-spunet-base/model/model_best.pth')


# weight = OrderedDict()

# for key, value in checkpoint["state_dict"].items():
#     if key.startswith("module."):
#         if comm.get_world_size() == 1:
#             key = key[7:]  # module.xxx.xxx -> xxx.xxx
#     else:
#         if comm.get_world_size() > 1:
#             key = "module." + key  # xxx.xxx -> module.xxx.xxx
#     weight[key] = value

# model.load_state_dict(weight)
model = model.cuda()
model.eval()

for i, b in enumerate(dataloader):

    with torch.no_grad():
        for key in b.keys():
            try:
                b[key] = b[key].cuda()
            except:
                pass
            
        pred = model(b)

    masks = pred['matched_masks']
    masks = torch.zeros((len(b['coord']), pred['matched_masks'][0].shape[1]))
    
    for i, g in enumerate(pred['matched_masks'][0]):
        masks[b['seg_indices'] == i] = g[None, :].cpu()
        
    #classes = pred['pred_masks']
    #print(classes)

    print(masks[0].shape, masks.argmax(-1).cpu().shape)

    data = VData.from_dict({
        'points': b['coord'].cpu(),
        'labels': masks.argmax(-1).cpu()
        # 'border_dist': pred['seg_logits'].cpu() 
    })

    o3d.io.write_point_cloud(f'data2_{i}.ply', data.to_o3d_pointcloud(color='labels'))
    exit(1)

    # input()    