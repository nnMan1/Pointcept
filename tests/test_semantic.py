import os
os.system("rm -r samples/*.npy")


from pointcept.utils.visualization import colors, to_o3d
import open3d as o3d
import numpy as np
from pointcept.datasets import ABCDataset, Assembly, Cetim, ScanNetDataset, Fuselage

ds = Fuselage(split='val_lr',
                data_root = 'data/fuselage/crops_250x250x250',
                classes=['body', 'body1', 'panel', 'rivets'],
                loop=1
    )

for d in ds:
    # try:
        pred = d['path'].replace('/', '_')

        pred = np.load(f'exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_grouping/result/{pred}_pred.npy')
    
        pcd = to_o3d(d['coord'], verts_colors=colors[7*pred+7])
        # pcd = to_o3d(d['coord'], verts_colors=colors[d['seg_indices'] % len(colors)])
        print(d['path'].replace('/', '_'), pred.shape, d['coord'].shape)
        print(np.asarray(pcd.colors).shape)
        o3d.io.write_point_cloud(f"exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_grouping/result/{d['path'].replace('/', '_')}.ply", pcd)
        # o3d.io.write_point_cloud(f"samples/{d['path'].replace('/', '_')}.ply", pcd)
        # o3d.visualization.draw_geometries([pcd])
    # except Exception as e:
    #     print(e)
    #     pass
