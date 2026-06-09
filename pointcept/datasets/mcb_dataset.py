import os
import glob
import h5py
import numpy as np
import torch
import trimesh
import os.path as osp
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from typing import Callable, List, Optional, Union

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .transform import Compose, TRANSFORMS
from .builder import DATASETS
from .transform import Compose, TRANSFORMS

class_names = [
        'Articulations, eyelets and other articulated joints',
        'Flanged plain bearings', 'Pulleys', 'Square', 'Bearing accessories',
        'Grooved pins', 'Radial contact ball bearings', 'Square nuts',
        'Bushes', 'Helical geared motors', 'Right angular gearings',
        'Standard fitting', 'Cap nuts', 'Hexagonal nuts', 'Right spur gears',
        'Studs', 'Castle nuts', 'Hinge', 'Rivet nut', 'Switch', 'Castor',
        'Hook', 'Roll pins', 'T-nut', 'Chain drives', 'Impeller', 
        'Screws and bolts with countersunk head', 'T-shape fitting', 'Clamps', 
        'Keys and keyways, splines', 'Screws and bolts with cylindrical head',
        'Taper pins', 'Collars', 'Knob', 'Screws and bolts with hexagonal head',
        'Tapping screws', 'Conventional rivets', 'Lever', 'Setscrew', 
        'Threaded rods', 'Convex washer', 'Locating pins', 'Slotted nuts', 
        'Thrust washers', 'Cylindrical pins', 'Locknuts', 'Snap rings', 
        'Toothed', 'Elbow fitting', 'Lockwashers', 'Socket', 'Turbine',
        'Eye screws', 'Nozzle', 'Spacers', 'Valve', 'Fan', 'Plain guidings', 
        'Split pins', 'Washer bolt', 'Flange nut', 'Plates, circulate plates', 
        'Spring washers', 'Wheel', 'Flanged block bearing', 'Plugs', 'Springs', 
        'Wingnuts'
    ]

@DATASETS.register_module()
class MCBDataset(Dataset):

    def __init__(
        self,
        split="train",
        data_root="data/mcb_dataset",
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
        class_names = class_names, 
        label_to_id=None,
    ):
        super(MCBDataset, self).__init__()
        self.data_root = data_root
        self.split = split

        self.class_names = class_names
        
        if label_to_id is not None:
            self.label_to_id = label_to_id
        else:
            self.label_to_id = {label: id for id, label in enumerate(class_names)}

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
        
        data_list = []

        class_names = ['*'] if 'other' in self.class_names else self.class_names

        if isinstance(self.split, str):
            for label in class_names:
                data_list += glob.glob(osp.join(self.data_root, self.split, label, '*.obj'))
        elif isinstance(self.split, Sequence):
            for split in self.split:
                for label in class_names:
                    data_list += glob.glob(osp.join(self.data_root, split, label, '*.obj'))
        else:
            raise NotImplementedError
        
        return data_list

    def get_data(self, idx):

        idx = idx % len(self.data_list)

        try:
            data = self.data_list[idx]
            pcd = trimesh.load_mesh(data).sample(4096, return_index=False) # weighted by face area by default
            coord = np.asarray(pcd)

            if len(coord) == 0:
                raise Exception("Number of points can not be 0")
        except:
            return self.get_data((idx + 1) % len(self.data_list))
        
        label = data.split('/')[-2]
        label_id = self.label_to_id[label] if label in self.class_names else self.label_to_id['other']
        
        return {
            'path': data,
            'coord': coord,
            'category': label_id,
            'path':  data
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
    
    def analyze_dataset(self):
        """
        Analyze the dataset to get the distribution of classes.
        """
        class_distribution = {label: 0 for label in self.class_names}
        for data in self.data_list:
            label = data.split('/')[-2]
            if label in class_distribution:
                class_distribution[label] += 1
            else:
                class_distribution['other'] += 1
        
        return class_distribution   

