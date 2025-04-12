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
import copy

@DATASETS.register_module("MechanicalAssemblySynth")
class MechanicalAssemblySynth(Dataset):

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
        augment_holes=False
    ):
        super(MechanicalAssemblySynth, self).__init__()
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

        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )
        
        # self.prepare_clustering()
        self.preloaded_data = [None for _ in self.data_list]
        self.augment_holes = augment_holes

        # for i in range(len(self.data_list)):
        #     self.get_data(i)

    # def prepare_clustering(self):
    #     for file in self.data_list:
    #         dir = os.path.dirname(file)

    #         with open(os.path.join(self.data_root, dir, 'annotations.json')) as json_file:
    #             annotations = json.load(json_file)

    #         if 'seg_indices' in annotations.keys():
    #             continue

    #         mesh = trimesh.load(f'{self.data_root}/{file}')
    #         vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
    #         faces = torch.from_numpy(mesh.faces.astype(np.int64))
    #         ind = segment_mesh(vertices, faces, 0.0001, 5).numpy()
            
    #         annotations['seg_indices'] = ind.tolist()

    #         with open(os.path.join(self.data_root, dir, 'annotations.json'), 'w') as json_file:
    #             json.dump(annotations, json_file)

    def get_data_list(self):
        
        if isinstance(self.split, str):
            data_list = open(os.path.join(self.data_root, f"{self.split}_files.txt")).readlines()
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += torch.load(open(os.path.join(self.data_root, f"{self.split}_files.txt")).readlines())
        else:
            raise NotImplementedError
        
        data_list = [f.strip() for f in data_list]   

        return data_list

    def get_data(self, idx):

        idx = idx % len(self.data_list)
        dir = self.data_list[idx]

        frames = sorted(glob.glob(os.path.join(self.data_root, dir, '*.ply')))
        annotations = sorted(glob.glob(os.path.join(self.data_root, dir, '*.json')))

        if self.preloaded_data[idx] != None:
            return copy.deepcopy(self.preloaded_data[idx])

        if os.path.exists(os.path.join(self.data_root, dir, 'cached.pth')):
            try:
                self.preloaded_data[idx] = torch.load(os.path.join(self.data_root, dir, 'cached.pth'))
                return copy.deepcopy(self.preloaded_data[idx])
            except Exception as e:
                print(f"Error loading {dir}: {e}")
                os.remove(os.path.join(self.data_root, dir, 'cached.pth'))
        
        vertices = []
        semantic_id = []
        instance_id = []
        frame_id = []

        try:
            for i, (annotation, frame) in enumerate(zip(annotations, frames)):
                pcd = trimesh.load(frame)
                vertices.append(pcd.vertices.astype(np.float32))
                labels = json.load(open(annotation))

                semantic_mapping = np.asarray([self.class_mapping[c] for c in labels['classes']])

                instance_id.append(labels['instance_id'])
                semantic_id.append(semantic_mapping[np.asarray(labels['semantic_id'])])
                frame_id.append(np.asarray([i] * len(labels['semantic_id'])))            

            vertices = np.concatenate(vertices, axis=0)
            instance_id = np.concatenate(instance_id, axis=0)
            semantic_id = np.concatenate(semantic_id, axis=0)
            frame_id = np.concatenate(frame_id, axis=0)
        except Exception as e:
            print(f"Error loading {dir}: {e}")
            return self.get_data(idx + 1)

        # Estimate normals for the point cloud
        mesh = trimesh.Trimesh(vertices=vertices, process=False)
        normals = mesh.vertex_normals
        

        if len(mesh.vertices) < 2048:
            del mesh
            return self.get_data(idx + 1)
    
        self.preloaded_data[idx] = {
            'coord': vertices,
            # 'coord': mesh.vertices[mask],
            # 'face': mesh.faces,
            'normal': normals,
            'instance': instance_id,
            'segment': semantic_id,
            'id': idx,
            'path': self.data_list[idx],
            # 'seg_indices': None,
            'frame_id': frame_id
        }

        torch.save(self.preloaded_data[idx], os.path.join(self.data_root, dir, 'cached.pth'))
        
        return copy.deepcopy(self.preloaded_data[idx])

    def get_data_name(self, idx):
        data_name = self.data_list[idx]
        data_name = data_name.replace(".ply", "")
        data_name = data_name.replace("/", "_")
        return data_name

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)    

        if self.augment_holes:
            data_dict = self.hole_augmentatior.remove_rivet(data_dict=data_dict)
            data_dict = self.hole_augmentatior.remove_rivet(data_dict=data_dict)
            data_dict = self.hole_augmentatior.remove_rivet(data_dict=data_dict)
            data_dict = self.hole_augmentatior.remove_rivet(data_dict=data_dict)
            data_dict = self.hole_augmentatior.remove_rivet(data_dict=data_dict)

        data_dict = self.transform(data_dict)

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
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

