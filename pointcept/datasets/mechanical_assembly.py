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

@DATASETS.register_module("MechanicalAssembly")
class MechanicalAssembly(Dataset):

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
        classes = []
    ):
        super(MechanicalAssembly, self).__init__()
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



    def prepare_clustering(self):
        for file in self.data_list:
            dir = os.path.dirname(file)

            with open(os.path.join(self.data_root, dir, 'annotations.json')) as json_file:
                annotations = json.load(json_file)

            if 'seg_indices' in annotations.keys():
                continue

            mesh = trimesh.load(f'{self.data_root}/{file}')
            vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
            faces = torch.from_numpy(mesh.faces.astype(np.int64))
            ind = segment_mesh(vertices, faces, 0.00001, 5).numpy()
            print(os.path.join(self.data_root, dir, 'annotations.json'))
            
            annotations['seg_indices'] = ind.tolist()

            with open(os.path.join(self.data_root, dir, 'annotations.json'), 'w') as json_file:
                json.dump(annotations, json_file)

    def get_hole_centers(self, data_dict):

        mask = np.where(data_dict['segment'] == self.class_mapping['hole'])[0]
        dbscan = DBSCAN(eps=5, min_samples=1)
        all_points = data_dict['coord']
        hole_points = all_points[mask]

        if len(hole_points) == 0:
            return data_dict

        dbscan.fit(hole_points)
        labels = dbscan.labels_

        centroids = []
        for label in set(labels):
            if label == -1:
                continue

            cluster_points = hole_points[labels == label]
            print(cluster_points.shape)
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

    
        centroids = np.array(centroids)
        distances = np.linalg.norm(centroids.T[:, None] - all_points[None, :].T, axis=0)
        close_points_mask = np.any(distances < 7, axis=1)
        data_dict['segment'][close_points_mask] = self.class_mapping['hole']
        
        return data_dict


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
        file = self.data_list[idx]
        dir = os.path.dirname(file)

        if self.preloaded_data[idx] != None:
            return self.preloaded_data[idx]
        

        with open(os.path.join(self.data_root, dir, 'annotations.json')) as json_file:
            annotations = json.load(json_file)

        hole_index = [i for i, x in enumerate(annotations['classes']) if x == 'hole'][0]


        if np.sum(np.asanyarray(annotations['semantic_id']) ==  hole_index) == 0: 
            return  self.get_data(idx + 1)

        mesh = trimesh.load(f'{self.data_root}/{file}')

        if len(mesh.vertices) < 2048:
            del mesh
            return self.get_data(idx + 1)
    
        # # Transform mesh to point cloud using uniform sampling to 30000 samples
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        point_cloud = np.asarray(o3d_mesh.sample_points_uniformly(number_of_points=250000).points)

        # Assign labels using KNN
        segment_labels = np.asarray(annotations['semantic_id'])
        mask = segment_labels != -1

        # Fit KNN on mesh vertices
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(mesh.vertices[segment_labels != -1])

        # # Find nearest neighbors for the sampled points
        distances, indices = knn.kneighbors(point_cloud)

        # Assign labels from the nearest neighbors
        classes = np.asarray([self.class_mapping[cls] for cls in annotations['classes']])
        instance_labels = np.asarray(annotations['instance_id'])[segment_labels != -1][indices.flatten()]
        normals =  mesh.vertex_normals[segment_labels != -1][indices.flatten()]
        seg_indices = np.asarray(annotations['seg_indices'])[segment_labels != -1][indices.flatten()]
        segment_labels = np.asarray(annotations['semantic_id'])[segment_labels != -1][indices.flatten()]
        segment_labels = classes[segment_labels]



        self.preloaded_data[idx] = self.get_hole_centers({
            'coord': point_cloud,
            # 'coord': mesh.vertices[mask],
            # 'face': mesh.faces,
            'normal': normals,
            'instance': instance_labels,
            'segment': segment_labels,
            'id': idx,
            'path': self.data_list[idx],
            'seg_indices': seg_indices
        })
        
        return self.preloaded_data[idx]

    def get_data_name(self, idx):
        data_name = self.data_list[idx]
        data_name = data_name.replace(".ply", "")
        data_name = data_name.replace("/", "_")
        return data_name

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
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

