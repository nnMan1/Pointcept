"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os

try:
    import cv2
    from PIL import Image
except Exception as e:
    print(e)    
try:
    import open3d as o3d
except Exception as e:
    print(e)

import numpy as np
import torch
from matplotlib import colors as mcolors
from pointcept.utils.registry import Registry
from sklearn.decomposition import PCA

colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())
colors = np.asarray([mcolors.to_rgba(color)[:3] for color in colors])
colors = colors[colors != np.asarray([1, 1, 1])].reshape([-1, 3])
colors = np.concatenate([colors[:5], colors[6:]], 0)

METRICS = Registry("visualizers")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x

class BaseVisualizer:

    def __init__(self, save_path, extension, logger=None):
        self.save_path = save_path
        self.extension = extension
        self.logger = logger

    def name_from_id(self, sample_id):
        return os.path.join(self.save_path, f'{sample_id}.{self.extension}')

    def __call__(self, data, sample_id):
        raise NotImplemented()

class PointCloudVisuzlizer(BaseVisualizer):

    def __call__(self, data, sample_id=None):
        assert 'coord' in data
        if 'color' in data:
            color = to_numpy(data['color'])
        else:
            if 'label' in data:
                color = colors[to_numpy(data['labels'])]
            else:
                color = np.ones_like(to_numpy(data['coord'])) * 0.5

        geom = o3d.geometry.PointCloud()
        geom.points = o3d.utility.Vector3dVector(to_numpy(data['coord']))
        geom.colors = o3d.utility.Vector3dVector(color)

        if sample_id is not None:
            o3d.io.write_point_cloud(self.name_from_id(sample_id), geom)

            if self.logger is not None:
                self.logger.info(f"Save Point Cloud to: {self.name_from_id(sample_id)}")

            
        return geom

class TrimeshVisualizer(BaseVisualizer):

    def __call__(self, data, sample_id=None):
        assert 'coord' in data
        assert 'faces' in data

        if 'color' in data:
            color = to_numpy(data['color'])
        else:
            if 'label' in data:
                color = colors[to_numpy(data['labels'])]
            else:
                color = np.ones_like(to_numpy(data['coord'])) * 0.5

        geom = o3d.geometry.TriangleMesh()
        geom.vertices = o3d.utility.Vector3dVector(to_numpy(data['coord']))
        geom.triangles = o3d.utility.Vector3dVector(to_numpy(data['faces']))
        geom.vertex_colors = o3d.utility.Vector3dVector(color)

        if sample_id is not None:
            o3d.io.write_triangle_mesh(self.name_from_id(sample_id), geom)

            if self.logger is not None:
                self.logger.info(f"Save Trimesh to: {self.name_from_id(sample_id)}")      
            
        return geom
    
class FeatureVisualizer(BaseVisualizer):

    def __init__(self, n_components, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

    def pca_features(self, features):
        """
        Visualize PCA features of point cloud.
        
        Args:
            coord (np.ndarray): Point cloud coordinates.
            features (np.ndarray): Features to visualize.
            file_path (str): Path to save the visualization.
            logger: Logger for logging information.
        Returns:
            o3d.geometry.PointCloud: Open3D point cloud object with PCA features.
        """

        pca = PCA(n_components=self.n_components)
        colors = pca.fit_transform(to_numpy(features))[:, -3:]

        colors -= colors.min(axis=0)
        colors /= colors.max(axis=0)
        
        return colors

class PointCloudFeatureVisualizer(PointCloudVisuzlizer, FeatureVisualizer):

    def __call__(self, data, sample_id=None):
        assert 'coord' in data
        assert 'feat' in data

        color = self.pca_features(data['feat'])

        return super()({'coord': data['coord'], 'color': color}, sample_id)

class TrimeshFeatureBisualizer(TrimeshVisualizer, FeatureVisualizer):

    def __call__(self, data, sample_id=None):
        assert 'coord' in data
        assert 'faces' in data
        assert 'feat' in data

        color = self.pca_features(data['feat'])

        return super()({'coord': data['coord'], 'faces': data['faces'], 'color': color}, sample_id)


#TODO: Replace PIL with cv2
class ImageVisualizer(BaseVisualizer):

    def __init__(self, resize=None, normalize=False, grayscale_color=[1, 1, 1], **kwargs):
        super().__init__(**kwargs)
        self.resize=resize
        self.normalize=normalize
        self.grayscale_color = grayscale_color

    def _to_rgb(self, image, grayscale=None):

        if grayscale is None:
            grayscale = self.grayscale_color

        grayscale = np.asanyarray(grayscale)

        if image.ndim == 2:
            return image[..., None].repeat(3, axis=2) * grayscale[None, None, ...]
        
        return image
    
    def __call__(self, data, sample_id=None, grayscale_color=None):

        assert 'image' in data

        image = to_numpy(data['image'])
        image = self._to_rgb(image)

        if self.resize is not None:
            interp = cv2.INTER_NEAREST if image.dtype == np.int32 or image.dtype == np.int16 else cv2.INTER_LINEAR
            image = cv2.resize(image, self.resize, interpolation=interp)
        
        if image.max() <= 1 or image.min() < 0:
            image = image * 255

        output_image = Image.fromarray(image.astype(np.uint8))

        if sample_id:
            output_image.save(self.name_from_id(sample_id))

            if self.logger is not None:
                self.logger.info(f"Save image to: {self.name_from_id(sample_id)}")    

        return output_image

class OverlayImageVisualizerr(ImageVisualizer):

    def __init__(self, alpha=0.35, grayscale_color=[(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1) ], **kwargs):
        super().__init__(grayscale_color=grayscale_color, **kwargs)
        self.alpha = alpha
    
    def __call__(self, data, sample_id=None):
        
        assert 'images' in data

        canvas = Image.new("RGBA", self.resize, (0, 0, 0, 0))

        for i, image in enumerate(data['images']):
            image = to_numpy(image)
            image = self._to_rgb(image, self.grayscale_color[i % len(self.grayscale_color)])

            if self.resize is not None:
                interp = cv2.INTER_NEAREST if image.dtype == np.int32 or image.dtype == np.int16 else cv2.INTER_LINEAR
                image = cv2.resize(image, self.resize, interpolation=interp)
            
            if image.max() <= 1 or image.min() < 0:
                image = image * 255

            image = Image.fromarray(image.astype(np.uint8))
            image = image.convert("RGBA")
            
            if i > 0:
                alpha_channel = image.getchannel('A').point(lambda p: p * self.alpha)
                image.putalpha(alpha_channel)
            else:
                image.putalpha(255)

            canvas = Image.alpha_composite(canvas, image)

        if sample_id:
            canvas.save(self.name_from_id(sample_id))

            if self.logger is not None:
                self.logger.info(f"Save image to: {self.name_from_id(sample_id)}")      

        return canvas

class StackImageVisualizer(ImageVisualizer):    
    def __call__(self, data, sample_id=None):
        
        assert 'images' in data

        images = []

        for i, image in enumerate(data['images']):
            image = to_numpy(image)
            image = self._to_rgb(image)

            if self.resize is not None:
                interp = cv2.INTER_NEAREST if image.dtype == np.int32 or image.dtype == np.int16 else cv2.INTER_LINEAR
                print(image.shape, self.resize)
                image = cv2.resize(image, self.resize, interpolation=interp)
            
            if image.max() <= 1 or image.min() < 0:
                image = image * 255

            images.append(image)

        images = np.concatenate(images, axis=1)
        canvas = Image.fromarray(images.astype(np.uint8))

        if sample_id:
            canvas.save(self.name_from_id(sample_id))

            if self.logger is not None:
                self.logger.info(f"Save image to: {self.name_from_id(sample_id)}")      

        return canvas

class ImageFeatureVisualizer(ImageVisualizer, FeatureVisualizer):

    def __call__(self, data, sample_id=None):
        assert 'feat' in data
        feat = data['feat']

        image_size = feat.shape[:2]

        image = self.pca_features(feat.reshape(-1, feat.shape[-1])).reshape(*image_size, -1)

        return super().__call__({'image': image}, sample_id)

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

            overlapped = torch.nonzero(iou > 0.5).cpu()
            keep[order[overlapped + i + 1]] = 0

    return torch.where(keep)[0]

