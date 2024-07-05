import open3d as o3d
from pointcept.datasets import Fuselage
from pointcept.utils.visualization import to_o3d, colors

ds = Fuselage()

for s in ds:
    print(s)
    s = to_o3d(s['coord'], verts_colors=colors[s['instance']%len(colors)])
    o3d.visualization.draw_geometries([s])
    exit(0)