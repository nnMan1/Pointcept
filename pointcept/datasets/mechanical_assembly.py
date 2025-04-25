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

class HoleAugmentor:

    def __init__(self, class_mapping):
        self.class_to_id = class_mapping

    def plane_interpolate(self, data_dict, interpolating_ids, k, center, radius, assign_sem_label, assign_ins_label):
        ret = {}

        points = data_dict['coord']

        a_ids, b_ids, alphas = [], [], []

        attp = 0
        
        while len(a_ids) < k: 
            a_id, b_id = np.random.choice(interpolating_ids, (2, ))
            alpha = np.random.uniform(0, 1)

            pt = alpha * points[a_id] + (1 - alpha) * points[b_id]
            if (np.linalg.norm(pt - center) < radius) | (attp == 50):
                attp = 0
                a_ids.append(a_id)
                b_ids.append(b_id)
                alphas.append(alpha)

            attp += 1

        a_ids = np.asarray(a_ids, dtype=np.int32)
        b_ids = np.asarray(b_ids, dtype=np.int32)
        alphas = np.asarray(alphas)[..., None]


        for key in ['coord', 'normal']:
            ret[key] = alphas * data_dict[key][a_ids] + (1 - alphas) * data_dict[key][b_ids]

        ret['segment'] = np.ones(k, np.int32) * assign_sem_label
        ret['instance'] = np.ones(k, np.int32) * assign_ins_label
        # ret['seg_indices'] = np.where(alphas[:, 0] < 0.5, data_dict['seg_indices'][a_ids],  data_dict['seg_indices'][b_ids])

        return ret

    def remove_radius(self,  data_dict, center, radius, label=-1):

        points, labels  = data_dict['coord'], data_dict['segment']
        dists =  np.linalg.norm(points - center, axis=-1)

        ids_keep= np.where(dists > radius) if label == -1 else np.where((dists > radius) | (labels != label))

        for key, val in data_dict.items():
            if isinstance(val, np.ndarray) and len(val) == len(points):
                data_dict[key]= data_dict[key][ids_keep]

        return data_dict
    
    def augment_rivet_hole_shape(self, data_dict, center, radius, direction):

        points, labels  = data_dict['coord'], data_dict['segment']
        dists =  np.linalg.norm(points - center, axis=-1)

        mask = dists < radius
        delta = np.exp(-np.random.uniform(0.01, 1)/(1 - (dists[mask] / radius + 1e-15) ** 2))

        direction = direction / np.linalg.norm(direction)
        direction += + np.random.normal(0, 0.2, 3)
        direction = direction / np.linalg.norm(direction)

        points[mask] += 3 * delta[..., None] * direction

        data_dict['coord'] = points

        return data_dict

    def remove_rivet(self, data_dict):
        
        rivet_id = self.class_to_id['rivet']
        hole_id = self.class_to_id['hole']

        if np.sum(data_dict['segment'] == rivet_id) == 0:
            return data_dict

        points, labels  = data_dict['coord'], data_dict['segment']
            
        ids = np.where(labels == rivet_id)[0]

        clusters = (
                        DBSCAN(
                            eps=0.95,
                            min_samples=1,
                            n_jobs=1,
                        )
                        .fit(points[ids])
                        .labels_
                    )
        
        cluster_id = np.random.choice(clusters, 1)[0]

        remove_ids = np.where(clusters == cluster_id)
        center = points[ids[remove_ids]].mean(axis=0)
        rivet_diam = np.linalg.norm(points[ids[remove_ids]] - center, axis=-1).max() * 1.3
        
        if rivet_diam > 10:
            return data_dict

        hole_diam_radius = max(rivet_diam + 3, 7)

        dists =  np.linalg.norm(points - center, axis=-1)
        ids_interesting = np.where(dists < hole_diam_radius)
        # ids_remove = np.logical_and(dists < hole_diam_radius, labels == rivet_id)
        ids_remove = dists < rivet_diam

        interpolating_ids = np.where((dists < hole_diam_radius) & (dists > rivet_diam))[0]

        if len(interpolating_ids) == 0:
            return data_dict

        # ids_remove = np.logical_and(ids_remove, np.cumsum(ids_remove) < np.random.uniform(0.9, 1) * ids_remove.sum())

        interpolated = (self.plane_interpolate(data_dict, interpolating_ids, ids_remove.sum(), center, rivet_diam, hole_id, 0))

        for key, val in data_dict.items():
            if isinstance(val, np.ndarray) and len(val) == len(points):
                data_dict[key][ids_remove] = interpolated[key]

        mask = np.where(dists < hole_diam_radius)
        labels[mask] = hole_id
        data_dict['instance'][mask] = data_dict['instance'].max() + 1

        data_dict['segment'] = labels


        if len(interpolated['coord']) == 0:
            return data_dict

        new_center = interpolated['coord'][np.linalg.norm(interpolated['coord'] - center, axis=-1).argmin()]

        if np.random.uniform() < 0.2:
            r = np.random.uniform(1, 2)
            data_dict = self.augment_rivet_hole_shape(data_dict, new_center, r, new_center - center)
            new_center = data_dict['coord'][np.linalg.norm(data_dict['coord'] - new_center, axis=-1).argmin()]
            data_dict = self.remove_radius(data_dict, new_center, 0.9*r)
        else:
            if np.random.uniform() < 0.7:
                data_dict = self.remove_radius(data_dict, new_center, np.random.uniform(1, 2))
            else:
                if np.random.uniform() < 0.5:
                    r = np.random.uniform(0, 0.5)
                    data_dict = self.augment_rivet_hole_shape(data_dict, new_center, r, new_center - center)

        return data_dict

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
        classes = [],
        augment_holes=False
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
        self.augment_holes = augment_holes
        self.hole_augmentatior = HoleAugmentor(self.class_mapping)

        # for i in range(len(self.data_list)):
        #     self.get_data(i)

    def prepare_clustering(self):
        for file in self.data_list:
            dir = os.path.dirname(file)
            print(os.path.join(self.data_root, dir, 'annotations.json'))

            with open(os.path.join(self.data_root, dir, 'annotations.json')) as json_file:
                annotations = json.load(json_file)

            if 'seg_indices' in annotations.keys():
                continue

            mesh = trimesh.load(f'{self.data_root}/{file}')
            vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
            faces = torch.from_numpy(mesh.faces.astype(np.int64))
            ind = segment_mesh(vertices, faces, 0.0001, 5).numpy()
            
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
        cls_instance = []
        for label in set(labels):
            if label == -1:
                continue

            cluster_points = hole_points[labels == label]
            print(cluster_points.shape)
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
            cls_instance.append(data_dict['instance'][mask][labels == label][0])

        centroids = np.array(centroids)
        cls_instance = np.array(cls_instance)
        distances = np.linalg.norm(centroids.T[:, None] - all_points[None, :].T, axis=0)
        close_points_mask = np.any(distances < 7, axis=1)
        close_points__nb = np.argmin(distances, axis=1)
        data_dict['segment'][close_points_mask] = self.class_mapping['hole']
        data_dict['instance'][close_points_mask] = cls_instance[close_points__nb[close_points_mask]]
        
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

        if os.path.exists(os.path.join(self.data_root, dir, 'cached.pth')):
            return torch.load(os.path.join(self.data_root, dir, 'cached.pth'))
                

        with open(os.path.join(self.data_root, dir, 'annotations.json')) as json_file:
            annotations = json.load(json_file)

        mesh = trimesh.load(f'{self.data_root}/{file}')

        if len(mesh.vertices) < 2048:
            del mesh
            return self.get_data(idx + 1)
    
        # # Transform mesh to point cloud using uniform sampling to 30000 samples
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        point_cloud = np.asarray(o3d_mesh.sample_points_uniformly(number_of_points=500000).points)

        # Assign labels using KNN
        segment_labels = np.asarray(annotations['semantic_id'])
        mask = segment_labels != -1

        # Fit KNN on mesh vertices
        try:
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(mesh.vertices[segment_labels != -1])
        except Exception as e:
            print(e)

        # # Find nearest neighbors for the sampled points
        distances, indices = knn.kneighbors(point_cloud)

        # Assign labels from the nearest neighbors
        classes = np.asarray([self.class_mapping[cls] for cls in annotations['classes']])
        instance_labels = np.asarray(annotations['instance_id'])[segment_labels != -1][indices.flatten()]
        normals =  mesh.vertex_normals[segment_labels != -1][indices.flatten()]
        seg_indices = np.asarray(annotations['seg_indices'])[segment_labels != -1][indices.flatten()]
        segment_labels = np.asarray(annotations['semantic_id'])[segment_labels != -1][indices.flatten()]
        segment_labels = classes[segment_labels]
        

        # Identify outlier points using DBSCAN
        dbscan = DBSCAN(eps=10, min_samples=5)
        dbscan.fit(point_cloud)
        outlier_mask = dbscan.labels_ == -1

        # Remove outlier points
        point_cloud = point_cloud[~outlier_mask]
        normals = normals[~outlier_mask]
        instance_labels = instance_labels[~outlier_mask]
        segment_labels = segment_labels[~outlier_mask]
        seg_indices = seg_indices[~outlier_mask]

        data = {
            'coord': point_cloud,
            # 'coord': mesh.vertices[mask],
            # 'face': mesh.faces,
            'normal': normals,
            'instance': instance_labels,
            'segment': segment_labels,
            'id': idx,
            'path': self.data_list[idx],
            'seg_indices': seg_indices
        }

        torch.save(data, os.path.join(self.data_root, dir, 'cached.pth'))
        
        return data

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

