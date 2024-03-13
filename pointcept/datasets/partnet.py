import os
import glob
import h5py
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from typing import Callable, List, Optional, Union

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class PartNet(Dataset):

    class_ids = {
            "Bowl-1": 0,
            "Faucet-1": 1,
            "Table-1": 2,
            "Vase-3": 3,
            "Lamp-3": 4,
            "Faucet-3": 5,
            "Vase-1": 6,
            "Door-2": 7,
            "Bed-3": 8,
            "Scissors-1": 9,
            "StorageFurniture-3": 10,
            "Bed-2": 11,
            "TrashCan-3": 12,
            "Bottle-3": 13,
            "Earphone-3": 14,
            "Chair-1": 15,
            "Door-1": 16,
            "Clock-3": 17,
            "StorageFurniture-1": 18,
            "Bag-1": 19,
            "Knife-3": 20,
            "Microwave-1": 21,
            "Laptop-1": 22,
            "Chair-2": 23,
            "Microwave-3": 24,
            "Bottle-1": 25,
            "Knife-1": 26,
            "Chair-3": 27,
            "Display-1": 28,
            "StorageFurniture-2": 29,
            "Microwave-2": 30,
            "Refrigerator-1": 31,
            "TrashCan-1": 32,
            "Bed-1": 33,
            "Table-3": 34,
            "Refrigerator-3": 35,
            "Keyboard-1": 36,
            "Earphone-1": 37,
            "Door-3": 38,
            "Lamp-1": 39,
            "Refrigerator-2": 40,
            "Dishwasher-3": 41,
            "Dishwasher-2": 42,
            "Table-2": 43,
            "Clock-1": 44,
            "Display-3": 45,
            "Mug-1": 46,
            "Lamp-2": 47,
            "Dishwasher-1": 48,
            "Hat-1": 49
        }

    seg_classes = {
        'Bowl-1': 4, 
        'Faucet-1': 8, 
        'Table-1': 11, 
        'Vase-3': 6, 
        'Lamp-3': 41, 
        'Faucet-3': 12, 
        'Vase-1': 4, 
        'Door-2': 4, 
        'Bed-3': 15, 
        'Scissors-1': 3, 
        'StorageFurniture-3': 24, 
        'Bed-2': 10, 
        'TrashCan-3': 11, 
        'Bottle-3': 9, 
        'Earphone-3': 10, 
        'Chair-1': 6, 
        'Door-1': 3, 
        'Clock-3': 11, 
        'StorageFurniture-1': 7, 
        'Bag-1': 4, 
        'Knife-3': 10, 
        'Microwave-1': 3, 
        'Laptop-1': 3, 
        'Chair-2': 30, 
        'Microwave-3': 6, 
        'Bottle-1': 6, 
        'Knife-1': 5, 
        'Chair-3': 39, 
        'Display-1': 3, 
        'StorageFurniture-2': 19, 
        'Microwave-2': 5, 
        'Refrigerator-1': 3, 
        'TrashCan-1': 5, 
        'Bed-1': 4, 
        'Table-3': 51, 
        'Refrigerator-3': 7, 
        'Keyboard-1': 3, 
        'Earphone-1': 6, 
        'Door-3': 5, 
        'Lamp-1': 18, 
        'Refrigerator-2': 6, 
        'Dishwasher-3': 7, 
        'Dishwasher-2': 5, 
        'Table-2': 42, 
        'Clock-1': 6, 
        'Display-3': 4, 
        'Mug-1': 4, 
        'Lamp-2': 28, 
        'Dishwasher-1': 3, 
        'Hat-1': 6
        }

    seg_levels = {
        'Bowl': [1], 
        'Faucet': [1, 3], 
        'Table': [1, 3, 2], 
        'Vase': [3, 1], 
        'Lamp': [3, 1, 2], 
        'Door': [2, 1, 3], 
        'Bed': [3, 2, 1], 
        'Scissors': [1], 
        'StorageFurniture': [3, 1, 2], 
        'TrashCan': [3, 1], 
        'Bottle': [3, 1], 
        'Earphone': [3, 1], 
        'Chair': [1, 2, 3], 
        'Clock': [3, 1], 
        'Bag': [1], 
        'Knife': [3, 1], 
        'Microwave': [1, 3, 2], 
        'Laptop': [1], 
        'Display': [1, 3], 
        'Refrigerator': [1, 3, 2], 
        'Keyboard': [1], 
        'Dishwasher': [3, 2, 1], 
        'Mug': [1], 
        'Hat': [1]
    }

    def __init__(
        self,
        split="train",
        data_root="data/Partnet",
        categories: Optional[Union[str, List[str]]] = None,
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        
        if categories is None:
            categories = list(self.seg_classes.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.seg_classes.keys() for category in categories)
        self.categories = categories
        self.__num_classes = 0

        super(PartNet, self).__init__()
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

            num_classes = 0
            data_list = []

            for category in sorted(self.categories):

                path = os.path.join(self.data_root, 'sem_seg_h5', category, f'{self.split.lower()}_files.txt')

                with open(path, 'r') as f:
                        filenames = [ name.strip() for name in f ] 

                for name in filenames:
                    data =  h5py.File(os.path.join(self.data_root, 'sem_seg_h5', category, name))

                    coords = np.asarray(np.array(data['data']))
                    seg_masks = np.array(data['label_seg'], dtype=np.int64)

                    seg_masks += num_classes

                    for coord, seg_mask in zip(coords, seg_masks):
                        data_list.append({
                            'coord': coord.T,
                            'seg_masks': seg_mask,
                            'category': category
                        })


                num_classes += self.seg_classes[category]

            return data_list

                    
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        coord = data["coord"]
        if "seg_masks" in data.keys():
            segment = data["seg_masks"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1

        cateogry = data['category']
       
        data_dict = dict(
            coord=coord.T,
            segment=segment,
            category=cateogry
        )

        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

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

# if __name__ == '__main__':

#     ds = PartNet(categories=['Earphone-1', 'Faucet-1'])

#     mc, mf, nc, nf = -1, -1, 1000, 1000

#     for i, t in enumerate(ds):
#         if t['category'] == 'Earphone-1':
#             mc = max(mc, t['segment'].max())
#             nc = min(nc, t['segment'].max())
#         else:
#             mf = max(mf, t['segment'].max())
#             nf = min(nf, t['segment'].max())

#         print(i, mc, nc, nf, mf, len(ds))

    # print(mc, mf)