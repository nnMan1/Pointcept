import os
import glob
import open3d as o3d
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from typing import Callable, List, Optional, Union

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from sklearn.cluster import DBSCAN

from .transform import Compose, TRANSFORMS
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class Fuselage(Dataset):

    def __init__(
        self,
        split="train",
        data_root="data/fuselage/crops",
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
        classes = ['body', 'body1', 'hole', 'panel', 'rivet', 'table'],
        augment_holes = False,
        merged_classes = None
    ):
        super(Fuselage, self).__init__()

        self.classes = classes
        self.class_to_id = {cls: i for (i, cls) in enumerate(classes)}

        if merged_classes is not None:
            for clss in merged_classes:
                for cls in clss[1:]:
                    self.class_to_id[cls] = self.class_to_id[clss[0]]   

        self.id_to_label = np.asarray([self.class_to_id[cls] for cls in self.classes])       

        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.augment_holes = augment_holes

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        
        self.data_list = self.get_data_list()

        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def plane_interpolate(self, data_dict, interpolating_ids, k, center, radius, assign_sem_label):
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
        ret['seg_indices'] = np.where(alphas[:, 0] < 0.5, data_dict['seg_indices'][a_ids],  data_dict['seg_indices'][b_ids])

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
        
        rivet_id = self.class_to_id['rivets']
        hole_id = self.class_to_id['hole']

        points, labels  = data_dict['coord'], data_dict['segment']
    
        if np.sum(labels == rivet_id) == 0:
            return data_dict
        
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
        rivet_diam = np.linalg.norm(points[ids[remove_ids]] - center, axis=-1).max() * 1.1

        hole_diam_radius = max(rivet_diam + 2, 3)

        dists =  np.linalg.norm(points - center, axis=-1)
        ids_interesting = np.where(dists < hole_diam_radius)
        # ids_remove = np.logical_and(dists < hole_diam_radius, labels == rivet_id)
        ids_remove = dists < rivet_diam

        interpolating_ids = np.where((dists < hole_diam_radius) & (dists > rivet_diam))[0]

        if len(interpolating_ids) == 0:
            return data_dict

        # ids_remove = np.logical_and(ids_remove, np.cumsum(ids_remove) < np.random.uniform(0.9, 1) * ids_remove.sum())

        interpolated = (self.plane_interpolate(data_dict, interpolating_ids, ids_remove.sum(), center, rivet_diam, hole_id))

        labels[np.where(dists < rivet_diam)] = hole_id

        data_dict['segment'] = labels

        for key, val in data_dict.items():
            if isinstance(val, np.ndarray) and len(val) == len(points):
                data_dict[key][ids_remove] = interpolated[key]

        if len(interpolated['coord']) == 0:
            return data_dict

        new_center = interpolated['coord'][np.linalg.norm(interpolated['coord'] - center, axis=-1).argmin()]

        # if np.random.uniform() < 0.7:
        #     data_dict = self.augment_rivet_hole_shape(data_dict, new_center, rivet_diam / 1.3, new_center - center)


        new_center = data_dict['coord'][np.linalg.norm(data_dict['coord'] - new_center, axis=-1).argmin()]

        if np.random.uniform() < 0.7:
            data_dict = self.remove_radius(data_dict, new_center, np.random.uniform(1, 3))

        return data_dict

    def remove_ranom_hole(self, data_dict):
        panel_id = self.class_to_id['panel']
        body_id = self.class_to_id['rivets']


        points, labels  = data_dict['coord'], data_dict['segment']

        ids = np.where((labels == panel_id) | (labels == body_id))[0]

        center_id = np.random.choice(ids, 1)[0]
        center = points[center_id]
        center_label = labels[center_id]
        
        data_dict = self.remove_radius(data_dict, center, np.random.uniform(2, 5), center_label)

        return data_dict

    def get_data_list(self):
        
        if isinstance(self.split, str):
            data_list_files = open(os.path.join(self.data_root, f"{self.split}_files.txt")).readlines()
        elif isinstance(self.split, Sequence):
            data_list_files = []
            for split in self.split:
                data_list_files += torch.load(open(os.path.join(self.data_root, f"{split}_files.txt")).readlines())
        else:
            raise NotImplementedError
        
        data_list_files = [f.strip() for f in data_list_files]

        data_list = []
        for file in data_list_files:
            print(self.data_root, file)
            data_list += open(os.path.join(self.data_root, file)).readlines()
        
        data_list = [f.strip() for f in data_list]

        return data_list

    def get_data(self, idx):

        idx = idx % len(self.data_list)

        data = self.data_list[idx]
        labels = []
        pcd = []

        for cls in self.classes:
            # print(os.path.join(self.data_root, data, cls, '*.ply'))
            for file in glob.glob(os.path.join(self.data_root, data, cls, '*.npy')):
                tmp = np.load(file)
                pcd.append(tmp)
                labels.append(np.ones(tmp.shape[0]) * self.class_to_id[cls])

        if len(pcd) == 0:
            return self.get_data(idx + 1)

        pcd = np.concatenate(pcd)
        labels = np.concatenate(labels)

        if len(pcd) < 2048:
            return self.get_data(idx + 1)

        div = 2

        seg_indices = pcd[::div, -1].astype(np.int32)
        uniq_seg_indices = np.unique(seg_indices)

        for i, ind in enumerate(uniq_seg_indices):
            seg_indices[seg_indices == ind] = i
            
        return {
            'coord': pcd[::div, :3],
            'segment': np.asarray(labels, dtype=np.int32)[::div],
            'normal': pcd[::div, 3:6],
            'seg_indices': seg_indices,
            'id': idx,
            'path': self.data_list[idx]
        } 

    def get_data_name(self, idx):
        return str(self.data_list[idx]).replace('/', '_')

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)

        # groups = data_dict['seg_indices']
        # group_size = np.bincount(groups)

        # for g, s in enumerate(group_size):
        #     if s > 0:
        #         labels = data_dict['segment']
        #         labels = labels[groups == g]
        #         label = np.bincount(labels).argmax()
        #         data_dict['segment'][groups == g] = label


        if self.augment_holes:
            data_dict = self.remove_rivet(data_dict)
            data_dict = self.remove_rivet(data_dict)
            # data_dict = self.remove_rivet(data_dict)
            # data_dict = self.remove_ranom_hole(data_dict)


        data_dict = self.transform(data_dict)

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        
        # segment = data_dict.pop("segment")
        segment = data_dict['segment']
        data_dict = self.transform(data_dict)
        data_dict_list = [data_dict]
        # for aug in self.aug_transform:
        #     data_dict_list.append(aug(deepcopy(data_dict)))

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

