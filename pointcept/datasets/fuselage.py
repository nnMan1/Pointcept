import os
import glob
<<<<<<< HEAD
import open3d as o3d
import numpy as np
import torch
=======
import h5py
import numpy as np
import json
>>>>>>> db8ad9c1e69143cd82a06689660abe8bfbd62b2e
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from typing import Callable, List, Optional, Union

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .transform import Compose, TRANSFORMS
from .builder import DATASETS
<<<<<<< HEAD
from .transform import Compose, TRANSFORMS
=======

import open3d as o3d
>>>>>>> db8ad9c1e69143cd82a06689660abe8bfbd62b2e

@DATASETS.register_module()
class Fuselage(Dataset):

    def __init__(
        self,
        split="train",
<<<<<<< HEAD
        data_root="data/fuselage/crops",
=======
        data_root="data/fuselage",
>>>>>>> db8ad9c1e69143cd82a06689660abe8bfbd62b2e
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
<<<<<<< HEAD
        classes = ['body', 'body1', 'hole', 'panel', 'riwet', 'table']
    ):
        super(Fuselage, self).__init__()

        self.classes = classes
        self.class_to_id = {cls: i for (i, cls) in enumerate(classes)}

=======
    ):
        super(Fuselage, self).__init__()
>>>>>>> db8ad9c1e69143cd82a06689660abe8bfbd62b2e
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

        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

<<<<<<< HEAD
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
=======
        with open(os.path.join(self.data_root, 'raw', 'labels.json')) as f:
            self.semantic_mapping = np.asarray(json.load(f))

    def get_data_list(self):
        dl = glob.glob(os.path.join(self.data_root, 'raw', '*', '*.npy'))

        if self.split == 'train':
            dl = dl[:int(0.8*len(dl))]
        else:
            dl =dl[int(0.8*len(dl)):]

        return dl
>>>>>>> db8ad9c1e69143cd82a06689660abe8bfbd62b2e

    def get_data(self, idx):

        idx = idx % len(self.data_list)

        data = self.data_list[idx]
<<<<<<< HEAD
        # print(os.path.join(self.data_root, 'scanns', data))

        labels = []
        pcd = o3d.geometry.PointCloud()

        for cls in self.classes:
            for file in glob.glob(os.path.join(self.data_root, data, cls, '*.ply')):
                tmp = o3d.io.read_point_cloud(file)
                pcd += tmp
                labels.append(np.ones(np.asarray(tmp.points).shape[0]) * self.class_to_id[cls])

        if(len(labels) == 0):
            return self.get_data(idx + 1)

        labels = np.concatenate(labels)
    
        return {
            'coord': np.asarray(pcd.points),
            'segment': labels.astype(np.int32),
=======
        # data = o3d.io.read_triangle_mesh(data).sample_points_uniformly(50000)
        data = np.load(data)
        
        return {
            'coord': data[::5, :3],
            'instance': data[::5, 3].astype(np.int64),
            'segment': self.semantic_mapping[data[::5, 3].astype(np.int64)],
>>>>>>> db8ad9c1e69143cd82a06689660abe8bfbd62b2e
            'id': idx,
            'path': self.data_list[idx]
        } 

    def get_data_name(self, idx):
<<<<<<< HEAD
        return str(self.data_list[idx]).replace('/', '_')
=======
        return str(idx)
>>>>>>> db8ad9c1e69143cd82a06689660abe8bfbd62b2e

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
<<<<<<< HEAD
        return len(self.data_list) * self.loop

=======
        return len(self.data_list) * self.loop
>>>>>>> db8ad9c1e69143cd82a06689660abe8bfbd62b2e
