from pointcept.datasets import build_dataset
from pointcept.models import build_model
from torch import Tensor
from pointcept.utils.visualization import pca_features_visualization


classes={"other": 0, 
        "gear": 0, 
        "nut": 0, 
        "screw": 0, 
        "axe": 0}

class_names = ["other"]

segment_ignore_index = ( -1, )

dataset = build_dataset(dict(
    type='HDF5_Dataset',
    split='train',
    data_root='data/data/processed/abc_dataset',
    load_images=True,
    load_features=dict(
        image_features='preextracted_features/dino_v2_small/features'
    ),
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
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.8),
            dict(type="RandomJitter", sigma=0.001, clip=0.02),
            dict(
                type="GridSample",
                grid_size=1,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
                return_grid_coord=True,
                keys=("coord", "normal", "segment", "instance"),
            ),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_segment",
                    "images",
                    "mappings_src",
                    "mappings_tgt",
                    "path",
                    "name",
                    "inverse",
                    "name",
                    "image_features"
                ),
                feat_keys=("coord"),
                offset_keys_dict=dict(
                    offset="coord", 
                    origin_offset="origin_coord", 
                    image_offset="images", 
                    mappings_offset="mappings_src",
                    instance_segment_offset="instance_segment"
                ),
            ),
        ],
        test_mode=False,
        classes=classes,
    ),
)

model = build_model(dict(
    type='Image2PointCLoud',
    return_features = ['feat'],
    project_fts = False
))

model.to('cuda')


for i, sample in enumerate(dataset):
    for k, v in sample.items():
        if isinstance(v, Tensor):
            sample[k] = v.cuda()
            
    fts = model(sample)
    pca_features_visualization(sample['coord'].cpu().numpy(), fts['feat'].cpu().numpy(), file_path=f"features{i}.ply")
    # print(sample['name'], sample['image_features'].shape, fts['feat'].shape, sample['coord'].shape)

    if i > 10:
        exit(0)