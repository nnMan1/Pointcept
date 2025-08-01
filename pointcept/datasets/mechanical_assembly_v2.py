import os
import glob
import json
import trimesh
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from typing import Callable, List, Optional, Union

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .transform import Compose, TRANSFORMS
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from sklearn.neighbors import NearestNeighbors
from segmentator import segment_mesh
from PIL import Image
import torchvision.transforms as transforms

from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRendererWithFragments, 
    MeshRasterizer, 
    SoftSilhouetteShader, 
    SoftPhongShader,
    MeshRenderer,
    BlendParams,
    TexturesUV,
    TexturesVertex
)

@DATASETS.register_module("MechanicalAssemblyV2")
class MechanicalAssemblyV2(Dataset):

    def __init__(
        self,
        split="train",
        data_root="data/assembly",
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
        classes = [],
        image_transform=None
    ):
        super(MechanicalAssemblyV2, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()

        if isinstance(classes, list):
            self.class_mapping = {c: i for i, c in enumerate(classes)}
        else:
            self.class_mapping = classes
        
        categories = list(self.class_mapping.items())
        categories.reverse()
        self.categories = {v: k for k,v in categories}

        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )
        
        # self.prepare_clustering()
        self.preloaded_data = [None for _ in self.data_list]

        self.image_size = (448, 448)  # or (518, 518) for ViT-Giant

        assert self.image_size[0] % 14 == 0 and self.image_size[1] % 14 == 0, \
            "Image size must be divisible by 14 for ViT models."

        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),  # or 518 for ViT-Giant
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # for i in range(len(self.data_list)):
        #     self.get_data(i)

    def get_mesh_name(self, dir):

        if os.path.exists(os.path.join(dir, "_simplified.stl")):
            return "_simplified.stl"

        mesh_files = glob.glob(os.path.join(dir, "*.ply")) + \
                     glob.glob(os.path.join(dir, "*.obj")) + \
                     glob.glob(os.path.join(dir, "*.stl"))
        if len(mesh_files) == 0:
            raise FileNotFoundError(f"No mesh file found in {dir}")
        return os.path.basename(mesh_files[0])

    def prepare_singe_clustering(self, file, annotations=None):

        dir = os.path.dirname(file)

        mesh = trimesh.load(file)
        vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
        faces = torch.from_numpy(mesh.faces.astype(np.int64))
        
        ind1 = segment_mesh(vertices, faces, 0.0001, 5).numpy()
        ind2 = segment_mesh(vertices, faces, 0.001, 10).numpy()
        ind3 = segment_mesh(vertices, faces, 0.001, len(vertices) // 5).numpy()
        
        if 'seg_indices' in annotations:
            annotations.pop('seg_indices')

        with open(os.path.join(dir, 'annotations.json'), 'w') as json_file:
            json.dump(annotations, json_file)
        
        groupings = {}
                        
        groupings['seg_indices1'] = ind1.tolist()
        groupings['seg_indices2'] = ind2.tolist()
        groupings['seg_indices3'] = ind3.tolist()

        with open(f'{dir}/grp.json', 'w') as json_file:
                json.dump(groupings, json_file)

        if os.path.exists(os.path.join(dir, 'cached.pth')):
            os.remove(os.path.join(dir, 'cached.pth'))

    def prepare_clustering(self):
        for dir in self.data_list:

            print(dir)

            if os.path.exists(os.path.join(dir, 'grp.json')):
                print(f"Clustering already prepared for {dir}")
                continue

            print(os.path.join(dir, 'annotations.json'))
            file = os.path.join(dir, self.get_mesh_name(dir))

            with open(os.path.join(dir, 'annotations.json')) as json_file:
                annotations = json.load(json_file)

            self.prepare_singe_clustering(file, annotations)

  
    def get_data_list(self):
        
        if isinstance(self.split, str):
            data_list = open(os.path.join(self.data_root, f"{self.split}_files.txt")).readlines()
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += torch.load(open(os.path.join(self.data_root, f"{self.split}_files.txt")).readlines())
        else:
            raise NotImplementedError

        data_list = [os.path.join(self.data_root, 'files', f.strip()) for f in data_list]

        return data_list
        
    def pixel_point_matches(
                self,
                pts_world: np.ndarray,
                faces: np.ndarray,  
                normals: np.ndarray,
                K: np.ndarray,                 # 3×4 projection matrix
                R: np.ndarray,                 # 3×3 rotation matrix
                T: np.ndarray,                 # 3D translation vector
                img_size: tuple[int, int],     # (H, W)
    ):
        
        mesh = Meshes(
            verts=[torch.from_numpy(pts_world.astype(np.float32))],
            faces=[torch.from_numpy(faces.astype(np.int64))],
        )

        cameras = FoVPerspectiveCameras(
            K=torch.from_numpy(K.astype(np.float32)).unsqueeze(0),
            R=torch.from_numpy(R.astype(np.float32)).unsqueeze(0),
            T=torch.from_numpy(T.astype(np.float32)).unsqueeze(0),
        )

        raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        fragments = rasterizer(mesh)
        mapping = fragments.pix_to_face[0, :, :, 0].numpy()  # (H, W)

        src = np.stack(np.where(mapping != -1)).T
        tgt = mapping[src[:, 0], src[:, 1]]

        tgt = faces[tgt].copy().reshape(-1)
        src = np.tile(src, (1, 3)).reshape(-1, 2)

        return src, tgt


    def get_data(self, idx):

        idx = idx % len(self.data_list)
        dir = self.data_list[idx]
        file = os.path.join(dir, self.get_mesh_name(dir))

        if self.cache and os.path.exists(os.path.join(dir, 'cached.pth')):
            return torch.load(os.path.join(dir, 'cached.pth'))
                

        with open(os.path.join( dir, 'annotations.json')) as json_file:
            annotations = json.load(json_file)

        with open(os.path.join(dir, 'grp.json')) as json_file:
            groups = json.load(json_file)

        mesh = trimesh.load(file)

        if len(mesh.vertices) < 2048:
            del mesh
            return self.get_data(idx + 1)
    
        # # Transform mesh to point cloud using uniform sampling to 30000 samples
        import open3d as o3d
        # o3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        # point_cloud = np.asarray(o3d_mesh.sample_points_uniformly(number_of_points=500000).points)

        # Assign labels using KNN
        segment_labels = np.asarray(annotations['semantic_id'])
        mask = segment_labels != -1

        # Fit KNN on mesh vertices
        # try:
        #     knn = NearestNeighbors(n_neighbors=1)
        #     knn.fit(mesh.vertices[segment_labels != -1])
        # except Exception as e:
        #     print(e)

        # # Find nearest neighbors for the sampled points
        # distances, indices = knn.kneighbors(point_cloud)

        # Assign labels from the nearest neighbors
        classes = np.asarray([self.class_mapping[cls] for cls in annotations['classes']])
        instance_labels = np.asarray(annotations['instance_id'])#[segment_labels != -1]#[indices.flatten()]
        normals =  mesh.vertex_normals.copy()#[segment_labels != -1]#[indices.flatten()]
        # seg_indices = np.asarray(annotations['seg_indices'])#[segment_labels != -1]#[indices.flatten()]
        segment_labels = np.asarray(annotations['semantic_id'])#[segment_labels != -1]#[indices.flatten()]
        segment_labels = classes[segment_labels]
        
        image_paths = sorted(glob.glob(os.path.join(dir, 'color', '*.png')))
        K_paths = sorted(glob.glob(os.path.join(dir,  'poses', '*K.txt')))
        R_paths = sorted(glob.glob(os.path.join(dir,  'poses', '*R.txt')))
        T_paths = sorted(glob.glob(os.path.join(dir,  'poses', '*T.txt')))
        mpappings_paths = sorted(glob.glob(os.path.join(dir,  'poses', '*mappings.npy')))
    
        images, mappings_src, mappings_tgt = [], [], []

        for i, (image_path, K, R, T) in enumerate(zip(image_paths, K_paths, R_paths, T_paths)):
            image = Image.open(image_path).convert('RGB')
            images.append(image)

            K = np.loadtxt(K)
            R = np.loadtxt(R)
            T = np.loadtxt(T)

            # K, R, t = self.decompose_camera_matrix(P)
            # print(f"Image {i}: K={K}, R={R}, t={t}")

            src, tgt = self.pixel_point_matches(
                pts_world=mesh.vertices,
                faces=mesh.faces,
                normals=normals,
                K=K,
                R=R,
                T=T,
                img_size=(448, 448)
            )

            src = np.stack([np.ones(len(src)) * i, src[:, 0], src[:, 1]], axis=1)  # (N, 3)
            
            mappings_src.append(src.astype(np.int32))
            mappings_tgt.append(tgt.astype(np.int32))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
            colors = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
            colors[tgt, 0] = 1 - colors[tgt, 0]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(f'image_{i}.ply', pcd)

        mappings_src = np.concatenate(mappings_src, axis=0)
        mappings_tgt = np.concatenate(mappings_tgt, axis=0)

        # Load all images from the directory
        # image_dir = os.path.join(self.data_root, dir)
        # image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
        # image_files = []
        # for ext in image_extensions:
        #     image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        # images = [open(img_path, 'rb').read() for img_path in image_files]

        # # Identify outlier points using DBSCAN
        # dbscan = DBSCAN(eps=10, min_samples=5)
        # dbscan.fit(point_cloud)
        # outlier_mask = dbscan.labels_ == -1

        # # Remove outlier points
        # point_cloud = point_cloud[~outlier_mask]
        # normals = normals[~outlier_mask]
        # instance_labels = instance_labels[~outlier_mask]
        # segment_labels = segment_labels[~outlier_mask]
        # seg_indices = seg_indices[~outlier_mask]

        data = {
            # 'coord': point_cloud,
            'coord':  deepcopy(mesh.vertices),
            'face': deepcopy(mesh.faces),
            'normal': normals,
            'instance': instance_labels,
            'segment': segment_labels,
            'id': idx,
            'path': self.data_list[idx],
            # 'seg_indices': seg_indices,
            'name': self.get_data_name(idx),
            'mappings_src': mappings_src,
            'mappings_tgt': mappings_tgt,
            'images': images,
            # 'ids': np.arange(len(mesh.vertices), dtype=np.int32),
        }

        for key in ['coord', 'normal', 'instance', 'segment']:
            if data[key].shape[0] != len(mesh.vertices):
                print(f"Warning: {key} shape mismatch in {file}: {data[key].shape[0]} != {len(mesh.vertices)}")

        data.update(groups)

        data['seg_indices'] = np.asarray(data['seg_indices1'])

        if self.cache:
            torch.save(data, os.path.join(self.data_root, dir, 'cached.pth'))
        
        return data

    def get_data_name(self, idx):
        data_name = self.data_list[idx]
        data_name = data_name.replace(".ply", "")
        data_name = data_name.replace(".obj", "")
        data_name = data_name.replace(".stl", "")
        data_name = data_name.replace("/", "_")
        return data_name

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)    

        if self.image_transform is not None:
            if 'images' in data_dict:
                data_dict['images'] = np.stack([
                    self.image_transform(image).numpy() for image in data_dict['images']
                ])
            else:
                data_dict['images'] = []

        data_dict = self.transform(data_dict)

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)

        if self.image_transform is not None:
            if 'images' in data_dict:
                data_dict['images'] = [
                    self.image_transform(image) for image in data_dict['images']
                ]
            else:
                data_dict['images'] = []

        data_dict_list = []        
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)

        for aug in self.aug_transform[:1]:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )

        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

