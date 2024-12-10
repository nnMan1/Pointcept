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

dataset = build_dataset(dict(
                    type='Assembly',
                        split="val",
                        transform=[
                            dict(type="CenterShift", apply_z=True),
                            dict(type="RBFunction", key="border_dist", gamma=0.3),
                            dict(
                                type="GridSample",
                                grid_size=0.3,
                                hash_type="fnv",
                                mode="train",
                                keys=("coord", "segment", "instance", "border_dist"),
                                return_grid_coord=True,
                            ),
                            dict(type="SphereCrop", point_max=100000, mode="random",
                                    keys=("coord", "grid_coord", "segment", "instance", "border_dist"),
                                    ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="ToTensor"),
                            dict(
                                type="Collect",
                                keys=("grid_coord", "coord", "segment","instance", "border_dist"),
                                feat_keys=("coord", ),
                            ),
                        ],
                        multiview=5,
                        test_mode=False,
                            ))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         collate_fn=partial(point_collate_fn),
                                        )

model = build_model(dict(
     type="EdgesDetector",
     backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=1,
        channels=(32, 64, 128, 128, 96, 96),
        layers=(2, 3, 4, 2, 2, 2),
    ),
    criteria=[
        dict(type="BCELoss", loss_weight=1.0)
    ]))

checkpoint = torch.load('exp/assembly/semseg-spunet-v1m1-0-base-rbf/model/model_last.pth')


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

for i, b in enumerate(dataset):

    if i % 50 != 0:
        continue

    with torch.no_grad():
        for key in b.keys():
            try:
                b[key] = b[key].cuda()
            except:
                pass
            
        pred = model(b)

    data = VData.from_dict({
        'points': b['coord'].cpu(),
        'border_dist': pred['seg_logits'].cpu() 
    })

    o3d.io.write_point_cloud(f'data2_{i}.ply', data.to_o3d_pointcloud(color='border_dist'))

    # input()    