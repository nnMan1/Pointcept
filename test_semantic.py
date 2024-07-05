import os
os.system("rm -r samples/*.npy")


from pointcept.utils.visualization import colors, to_o3d
import open3d as o3d
import numpy as np
from pointcept.datasets import ABCDataset, Assembly, Cetim, ScanNetDataset, Fuselage

ds = Fuselage(split='val',
                data_root = 'data/fuselage/crops_250x250x250',
              transform=[
                    dict(type="CenterShift", apply_z=True),
                    # dict(type="NormalizeColor"),
                ],
                classes=['body', 'body1', 'panel', 'riwet']
    )

for d in ds:
    try:
        pred = d['path'].replace('/', '_')
        pred = np.load(f'exp/fuselage/semseg-spunet-v1m1-0-base_250x250x250_hard_rot/result/{pred}_pred.npy')
    
        pcd = to_o3d(d['coord'], verts_colors=colors[6*pred+5])
        print(pred.shape, d['coord'].shape)
        print(np.asarray(pcd.colors).shape)
        o3d.io.write_point_cloud(f"samples/{d['path'].replace('/', '_')}.ply", pcd)
        o3d.visualization.draw_geometries([pcd])
    except:
        pass
