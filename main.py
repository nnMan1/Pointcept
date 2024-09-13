from pointcept.utils import visualization
from pointcept.datasets import Fuselage, ABCDataset, Assembly, ScanNetDataset
import open3d as o3d
import torch_scatter

class_names = [
    "assembly",
]
num_classes = 1
segment_ignore_index = (-1, )

ds = Assembly(
        split="train",
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                ),
                feat_keys=("coord"),
            ),
        ],
        test_mode=False,
    )


for d in ds:

    # id += 1

    # if (torch_scatter.scatter_min(d['segment'], d['seg_indices'])[0] - torch_scatter.scatter_min(d['segment'], d['seg_indices'])[0]).min() != 0:
    #     raise Exception()

    # print(id)

    coords = d['coord']
    # seg = d['seg_indices'] % len(visualization.colors)
    pcd = visualization.to_o3d(coords)
    o3d.io.write_point_cloud('pcd.ply', pcd)
    # o3d.visualization.draw_geometries([pcd])
    exit(0)
    