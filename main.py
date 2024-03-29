import time
import torch
from pointcept.datasets.builder import build_dataset
from pointcept.models.mask_3d import Mask3D

config = dict(
    type="Mask-3D",
    backbone=dict(
        type="MinkUNet34C",
        in_channels = 3,
        out_channels = 0,
        out_fpn=True, #return intermidiate features
    ),
    position_encoding=dict(
        type='PositionEmbeddingCoordsSine',
        pos_type="fourier",
        d_pos=96,
        gauss_scale=1,
        normalize=True,
    ),
    mask_module_config=dict(
        num_classes=1, 
        return_attn_masks=True, 
        use_seg_masks=False
    ),
    num_decoders=1,
    dim_feedforward=1024,
    hidden_dim=128,
    mask_dim=128
)

model = Mask3D( 
                backbone=dict(
                    type="MinkUNet34C",
                    in_channels = 3,
                    out_channels = 128,
                    out_fpn=True, #return intermidiate features
                ),
                position_encoding=dict(
                    type='PositionEmbeddingCoordsSine',
                    pos_type="fourier",
                    d_pos=128,
                    gauss_scale=1,
                    normalize=True,
                ),
                mask_module_config=dict(
                    num_classes=1, 
                    return_attn_masks=True, 
                    use_seg_masks=False
                ),
                query_refinement_config=dict(
                    pre_norm=False,
                    num_heads=8, 
                    dropout=0
                ),
                num_decoders=1,
                dim_feedforward=1024,
                hidden_dim=128,
                mask_dim=128).cuda()
# print(model)

# from pointcept.models.sparse_unet.mink_unet import MinkUNet34C

# model = MinkUNet34C(3, 128).to('cuda')

from pointcept.datasets.transform import GridSample
import numpy as np

pcd1 = {
    'coord': np.random.random([2048, 3]),
}

pcd2 = {
    'coord': np.random.random([2048, 3]),
}

gs = GridSample(grid_size=0.02,
                keys=('coord', ),
                return_grid_coord=True)



pcd1 = gs(pcd1)
pcd2 = gs(pcd2)

data = {
    'coord': torch.cat([torch.tensor(pcd1['coord']), torch.tensor(pcd2['coord'])]).to('cuda').float(),
    'grid_coord': torch.cat([torch.tensor(pcd1['grid_coord']), torch.tensor(pcd2['grid_coord'])]).to('cuda').int(),
    'offset': torch.tensor([len(pcd1['coord']), len(pcd1['coord'])+len(pcd2['coord'])], dtype=torch.int64).cuda(),
    'seed_ids': torch.stack([torch.randint(0, len(pcd1['coord']), (100,)), torch.randint(0, len(pcd2['coord']), (100,))])
}

# for key in data.keys():
#     print(key, data[key].shape, data[key].dtype)

data['feat'] = data['coord']


t = time.time()
out = model(data)

for key in out.keys():
    print(key, out[key].shape, out[key].dtype)

print(time.time() - t)

# train=dict(
#         type='Assembly',
#         split="train",
#         data_root='data/assembly',
#         transform=[
#             dict(type="CenterShift", apply_z=True),
#             dict(
#                 type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
#             ),
#             # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
#             dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
#             dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
#             dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
#             dict(type="RandomScale", scale=[0.9, 1.1]),
#             # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
#             dict(type="RandomFlip", p=0.5),
#             dict(type="RandomJitter", sigma=0.005, clip=0.02),
#             # dict(type="ElasticDistortion", distortion_params=[[2, 4], [8, 16]]),
#             dict(type="FPSSeed", n_points=100),
#             dict(
#                 type="GridSample",
#                 grid_size=0.5,
#                 hash_type="fnv",
#                 mode="train",
#                 return_grid_coord=True,
#                 keys=("coord", "segment", "instance"),
#             ),
#             dict(type="SphereCrop", sample_rate=0.8, mode="random"),
#             dict(
#                 type="InstanceParser",
#                 segment_ignore_index=(-1, ),
#                 instance_ignore_index=-1,
#             ),
#             dict(type="ToTensor"),
#             dict(
#                 type="Collect",
#                 keys=(
#                     "coord",
#                     "grid_coord",
#                     "segment",
#                     "instance",
#                     "instance_centroid",
#                     "bbox",
#                     "seed_ids"
#                 ),
#                 feat_keys=("grid_coord"),
#             ),
#         ],
#         test_mode=False,
#     )

# dataset = build_dataset(train)

# for d in dataset:
#     print(d.keys())
#     exit(0)