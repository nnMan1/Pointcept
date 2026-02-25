import os
import io
import glob
import h5py
import numpy as np
import torch
import pickle
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
# from segmentator import segment_mesh
from PIL import Image
import torchvision.transforms as transforms

@DATASETS.register_module()
class HDF5_Dataset(Dataset):

    def __init__(
        self,
        split="train",
        data_root="data/assembly",
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        classes = [],
        load_images=True,
        image_size=(448, 448),
        image_transform=None,
        load_features={},
        loop=1,
    ):
        super(HDF5_Dataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop
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

        self.image_size = image_size 

        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),  # or 518 for ViT-Giant
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.load_images = load_images

        self.open_files = {}

        self.open_features_files = {}
        self.load_features = load_features

        for k, v in self.load_features.items():
            self.load_features[k] = {}

            with open(os.path.join(v, 'feature_index.pkl'), 'rb') as f:
                index_map = pickle.load(f)

            for u, i, uid in index_map:
                self.load_features[k][uid] = f'{v}/{os.path.basename(u)}'

        
    def get_data_list(self):
        
        if isinstance(self.split, str):
            data_list = open(os.path.join(self.data_root, f"{self.split}_files.txt")).readlines()
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += torch.load(open(os.path.join(self.data_root, f"{self.split}_files.txt")).readlines())
        else:
            raise NotImplementedError

        self.data_list = []
        self.h5_files = [os.path.join(self.data_root, f.strip()) for f in data_list]
        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                for sample_id in f.keys():
                    self.data_list.append((h5_file, sample_id))

        return self.data_list

    def get_data(self, idx):

        h5_path, sample_id = self.data_list[idx % len(self.data_list)]

        if h5_path not in self.open_files:
            if len(self.open_files) > 32: 
                oldest_path = next(iter(self.open_files))
                self.open_files[oldest_path].close()
                del self.open_files[oldest_path]
            self.open_files[h5_path] = h5py.File(h5_path, 'r', swmr=True)

        f = self.open_files[h5_path]
        sample = f[sample_id]
        
        mesh_normals = np.asarray(sample['mesh_face_normals'])
        
        data = {
            'coord': [],
            'normal': [],
            'segment': [],
            'instance': [],
            'images': [],            
            'mappings_src': [],
            'mappings_tgt': [],
            'name': f'{os.path.basename(h5_path)}#{sample_id}',
            'path': h5_path
        }

        if self.load_images:
            point_offset = 0
            view_keys = sorted([k for k in sample.keys() if k.startswith('view_')])
            
            for i, v_key in enumerate(view_keys):
                view = sample[v_key]
                
                png_bytes = view['image_png'][:]
                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                w, h = img.size
                
                p2p = np.array(view['p2p']) 
                p2f = np.array(view['p2f']) 
                            
                point2face = p2f.flatten()[p2p]
                mask = point2face < len(mesh_normals)
       

                point2face = point2face[mask]
                
                coords = np.array(view['pc']).reshape(-1, 3)
        
                data['images'].append(img)
                data['coord'].append(coords[mask])
                data['normal'].append(mesh_normals[point2face])
                data['segment'].append(np.array(view['point_semantic'])[mask])
                data['instance'].append(np.array(view['point_instance'])[mask])

                n_points = mask.sum()
            
                src = np.stack([
                    np.full(n_points, i, dtype=np.int32), 
                    p2p[mask] // w,                            
                    p2p[mask] % w                              
                ]).T

                tgt = np.arange(point_offset, point_offset + n_points)
                data['mappings_src'].append(src)
                data['mappings_tgt'].append(tgt)

                point_offset += n_points

        for key in ['coord', 'normal', 'segment', 'instance', 'mappings_src', 'mappings_tgt']:
            if len(data[key]) > 0:
                data[key] = np.concatenate(data[key], axis=0)
        
        keep_ids = np.arange(len(data['coord']))
        while len(keep_ids) > 400000:
            keep_ids = keep_ids[::2]

        for key in ['coord', 'normal', 'segment', 'instance', 'mappings_src']:
            data[key] = data[key][keep_ids]

        data['mappings_tgt'] = np.arange(len(data['mappings_src']))

        while np.linalg.norm(data['coord'].max(axis=0) - data['coord'].min(axis=0)) < 80:
            data['coord'] *= 2

        while np.linalg.norm(data['coord'].max(axis=0) - data['coord'].min(axis=0)) > 400:
            data['coord'] /= 2

        if len(keep_ids) < 1000:
            return self.get_data(idx + 1)

        for ftk_key in self.load_features:
            h5_path = self.load_features[ftk_key][data['name']]
            
            if h5_path not in self.open_features_files:
                if len(self.open_files) > 32: 
                    oldest_path = next(iter(self.open_features_files))
                    self.open_features_files[oldest_path].close()
                    del self.open_features_files[oldest_path]
                self.open_features_files[h5_path] = h5py.File(h5_path, 'r', swmr=True)

            f = self.open_features_files[h5_path]
            data[ftk_key] = np.asarray(f[data['name']]['features'])

            # print("Loaded features for", data['name'], "with shape", data[ftk_key].shape, data['mappings_src'].shape, np.concatenate(data['coord'], axis=0).shape)
            # data[ftk_key] = data[ftk_key][data['mappings_src'][:, 0], data['mappings_src'][:, 1] // 16, data['mappings_src'][:, 2] // 16] # Map from image pixel to point
            # print("Loaded features for", data['name'], "with shape", data[ftk_key].shape)
            # data[ftk_key] = data[ftk_key][keep_ids]

        return data     

    def prepare_train_data(self, idx):
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

