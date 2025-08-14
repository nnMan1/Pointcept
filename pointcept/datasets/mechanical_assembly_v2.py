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

    def get_mesh_name(self, dir):

        mesh_files = glob.glob(os.path.join(dir, "*.ply")) + \
                     glob.glob(os.path.join(dir, "*.obj")) + \
                     glob.glob(os.path.join(dir, "*.stl"))
        if len(mesh_files) == 0:
            raise FileNotFoundError(f"No mesh file found in {dir}")
        return os.path.basename(mesh_files[0])

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

    def get_data(self, idx):

        idx = idx % len(self.data_list)
        dir = self.data_list[idx]
        file = os.path.join(dir, self.get_mesh_name(dir))

        if self.cache and os.path.exists(os.path.join(dir, 'cached.pth')):
            return torch.load(os.path.join(dir, 'cached.pth'))
                
        with open(os.path.join( dir, 'annotations.json')) as json_file:
            annotations = json.load(json_file)

        if 'semantic_id' not in annotations:
            annotations['semantic_id'] = np.zeros_like(annotations['instance_id'])

        with open(os.path.join(dir, 'grp_1e-05_100.json')) as json_file:
            groups = np.asarray(json.load(json_file))

        mesh = trimesh.load(file)

        if len(mesh.vertices) < 2048:
            del mesh
            return self.get_data(idx + 1)
    
        segment_labels = np.asarray(annotations['semantic_id'])

        classes = np.asarray([self.class_mapping[cls] for cls in annotations['classes']])
        instance_labels = np.asarray(annotations['instance_id'])
        normals =  mesh.vertex_normals.copy()
        segment_labels = np.asarray(annotations['semantic_id'])
        segment_labels = classes[segment_labels]
        
        image_paths = sorted(glob.glob(os.path.join(dir, 'color', '*.png')))
        K_paths = sorted(glob.glob(os.path.join(dir,  'poses', '*K.txt')))
        R_paths = sorted(glob.glob(os.path.join(dir,  'poses', '*R.txt')))
        T_paths = sorted(glob.glob(os.path.join(dir,  'poses', '*T.txt')))
        mapping_paths = sorted(glob.glob(os.path.join(dir,  'poses', '*mapping.txt')))
    
        images, mappings_src, mappings_tgt = [], [], []

        for i, (image_path, K, R, T, mapping) in enumerate(zip(image_paths, K_paths, R_paths, T_paths, mapping_paths)):
            image = Image.open(image_path).convert('RGB')
            images.append(image)

            K = np.loadtxt(K)
            R = np.loadtxt(R)
            T = np.loadtxt(T)

            mapping = np.loadtxt(mapping, dtype=np.int32)
            src = np.stack(np.where(mapping != -1)).T
            tgt = mapping[src[:, 0], src[:, 1]]            
            tgt = mesh.faces[tgt].copy().reshape(-1)
            src = np.tile(src, (1, 3)).reshape(-1, 2)

            src = np.stack([np.ones(len(src)) * i, src[:, 0], src[:, 1]], axis=1)  # (N, 3)
            
            mappings_src.append(src.astype(np.int32))
            mappings_tgt.append(tgt.astype(np.int32))


        mappings_src = np.concatenate(mappings_src, axis=0)
        mappings_tgt = np.concatenate(mappings_tgt, axis=0)


        data = {
            'coord':  deepcopy(mesh.vertices),
            'face': deepcopy(mesh.faces),
            'normal': normals,
            'instance': instance_labels,
            'segment': segment_labels,
            'id': idx,
            'path': self.data_list[idx],
            'name': self.get_data_name(idx),
            'mappings_src': mappings_src,
            'mappings_tgt': mappings_tgt,
            'images': images,
            'seg_indices': groups
        }

        for key in ['coord', 'normal', 'instance', 'segment']:
            if data[key].shape[0] != len(mesh.vertices):
                print(f"Warning: {key} shape mismatch in {file}: {data[key].shape[0]} != {len(mesh.vertices)}")

        if self.cache:
            torch.save(data, os.path.join(dir, 'cached.pth'))
        
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

