import os
import glob
import h5py
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
class Assembly(Dataset):

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
    ):
        super(Assembly, self).__init__()
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
            data_list = open(os.path.join(self.data_root, 'raw', f"{self.split}_files.txt")).readlines()
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += torch.load(open(os.path.join(self.data_root, 'raw', f"{self.split}_files.txt")).readlines())
        else:
            raise NotImplementedError
        
        data_list = [f.strip() for f in data_list]
        dl = []

        for i in range(6):
            dl += [f'{f}_{i}.h5' for f in data_list]

        return dl

    def get_data(self, idx):

        idx = idx % len(self.data_list)

        data = self.data_list[idx]
        data = h5py.File(os.path.join(self.data_root, 'raw', data),'r')
        return {
            'coord': np.asarray(data['coord'])[::3],
            'instance': np.asarray(data['instance'])[::3],
            'segment': np.zeros(np.asarray(data['coord']).shape[0], dtype=np.int32)[::3],
            'id': idx,
            'path': self.data_list[idx]
        } 

    def get_data_name(self, idx):
        return str(self.data_list[idx].assembly)

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

