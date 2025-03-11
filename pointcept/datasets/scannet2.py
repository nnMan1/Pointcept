"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)

import torch_scatter


@DATASETS.register_module()
class ScanNetDataset2(Dataset):
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        transform=None,
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
        ignore_semantic = (-1, 0, 1),
        ignore_instance = (-1, )
    ):
        super(ScanNetDataset2, self).__init__()
        self.ignore_semantic = ignore_semantic
        self.ifnore_instance = ignore_instance
        self.data_root = data_root
        self.sufix = '_inst_nostuff.pth'
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

        if lr_file:
            self.data_list = [
                os.path.join(data_root, "train", name + ".pth")
                for name in np.loadtxt(lr_file, dtype=str)
            ]
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):

        filenames = glob.glob(os.path.join(self.data_root, self.split, '*' + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        # filenames = filenames[:12]
        return filenames

    def get_data(self, idx):

        filename = self.data_list[idx]

        if self.with_label:
            data = torch.load(filename)
        else:
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            data = xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label
        
        data_dict = dict(
            coord=data[0],
            color=data[1],
            segment=segment,
            instance=instance,
            scene_id=scene_id,
            seg_indices=seg_indices,
            group_segment=group_segment
        )

        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            try:
                data = torch.load(data_path)
            except:
                print(data_path)
                raise Exception(data_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)

        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        seg_indices = data["seg_indices"]
        
        if "semantic_gt20" in data.keys():
            segment = data["semantic_gt20"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1

        uni = np.unique(seg_indices)

        # #TODO: OVO NIJE DOBRO
        segment = segment - 2
        segment[segment < 0] = -1

        for i, v in enumerate(uni):
            seg_indices[seg_indices == v] = i

        group_segment = torch_scatter.scatter_mean(torch.tensor(segment), seg_indices).numpy()

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
            seg_indices=seg_indices,
            group_segment=group_segment
        )

        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(segment).astype(np.bool)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
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


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    class2id = np.array(VALID_CLASS_IDS_200)

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt200" in data.keys():
            segment = data["semantic_gt200"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            segment[sampled_index] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict


if __name__ == '__main__':
    ds = ScanNetDataset()
    print(len(ds))