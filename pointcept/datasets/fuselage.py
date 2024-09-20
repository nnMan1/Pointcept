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
        classes = ['body', 'body1', 'hole', 'panel', 'rivet', 'table']
    ):
        super(Fuselage, self).__init__()

        self.classes = classes
        self.class_to_id = {cls: i for (i, cls) in enumerate(classes)}

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

        groups = data_dict['seg_indices']
        group_size = np.bincount(groups)
        soft_gorups = data_dict['seg_indices']

        for g, s in enumerate(group_size):
            if s > 0:
                labels = data_dict['segment']
                labels = labels[groups == g]
                label = np.bincount(labels).argmax()
                data_dict['segment'][groups == g] = label

        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        # segment = data_dict.pop("segment")
        segment = data_dict['segment']
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

