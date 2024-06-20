"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import open3d as o3d
import numpy as np
import torch
from matplotlib import colors as mcolors

colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())
colors = np.asarray([mcolors.to_rgba(color)[:3] for color in colors])
colors = colors[colors != np.asarray([1, 1, 1])].reshape([-1, 3])

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(coord) if color is None else color
    )
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def save_bounding_boxes(
    bboxes_corners, color=(1.0, 0.0, 0.0), file_path="bbox.ply", logger=None
):
    bboxes_corners = to_numpy(bboxes_corners)
    # point list
    points = bboxes_corners.reshape(-1, 3)
    # line list
    box_lines = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    lines = []
    for i, _ in enumerate(bboxes_corners):
        lines.append(box_lines + i * 8)
    lines = np.concatenate(lines)
    # color list
    color = np.array([color for _ in range(len(lines))])
    # generate line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Boxes to: {file_path}")


def save_lines(
    points, lines, color=(1.0, 0.0, 0.0), file_path="lines.ply", logger=None
):
    points = to_numpy(points)
    lines = to_numpy(lines)
    colors = np.array([color for _ in range(len(lines))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Lines to: {file_path}")


def nms(masks: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:

    # masks = to_numpy(masks)
    # scores = to_numpy(scores)
    
    order = torch.argsort(-scores)
    indices = torch.arange(masks.shape[-1])
    keep = torch.ones_like(indices, dtype=torch.bool, device=masks.device)
    
    masks = 1 / (1 + torch.exp(-masks))
    masks[masks > 0.5] = 1
    masks[masks <= 0.5] = 0
    
    for i in indices:
        if keep[order[i]]:
            mask = masks[:, order[i]]
            inter = (mask[:, None] * masks[:,  order]).sum(0)
            union = torch.logical_or(mask[:, None], masks[:,  order]).sum(0)
            iou = inter / union
            iou = iou[i+1:]

            overlapped = torch.nonzero(iou > iou_threshold).cpu()
            keep[order[overlapped + i + 1]] = 0

    return torch.where(keep)[0]


def to_o3d(pos, faces=None, verts_colors=None):

    if faces is not None:
        geom = o3d.geometry.TriangleMesh()
        geom.vertices = o3d.utility.Vector3dVector(to_numpy(pos))
        geom.triangles = o3d.utility.Vector3iVector(to_numpy(faces))
    else:
        geom = o3d.geometry.PointCloud()
        geom.points = o3d.utility.Vector3dVector(to_numpy(pos))

        if verts_colors is not None:
            geom.colors = o3d.utility.Vector3dVector(to_numpy(verts_colors))

    return geom