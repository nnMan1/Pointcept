"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import h5py
import time
import numpy as np
from uuid import uuid4

from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from functools import partial

from .defaults import worker_init_fn, create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)
try:
    import pointops
except:
    pointops = None


PREEXTRACTORS = Registry("feature_preextractor")


class ExtractorBase:
    def __init__(self, cfg, model=None, data_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "feature_preextracting.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        print(cfg)
        self.base_path = os.path.join(cfg.save_path, "features")

        if not os.path.exists(self.base_path):
            make_dirs(self.base_path)

        self.rank = comm.get_rank()
        self.h5_file = None
        self.current_shard_idx = 0
        self.samples_in_current_shard = 0
        self.shard_size = cfg.shard_size
        self._get_new_shard()
    
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if data_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.data_loader = self.build_data_loader()
        else:
            self.data_loader = data_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )

        if self.cfg.weight is not None and os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        
        return model

    def build_data_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=True,
        )
        return train_loader

    def extract(self):
        raise NotImplementedError

    def _get_new_shard(self):
        if self.h5_file:
            self.h5_file.close()

        shard_path = f"{self.base_path}/{self.rank}_shard_{self.current_shard_idx:04d}.h5"
        print(shard_path)
        self.h5_file = h5py.File(shard_path, 'w', libver='latest')
        self.current_shard_idx += 1
        self.samples_in_current_shard = 0
        
        self.dt_float = h5py.vlen_dtype(np.dtype('float32'))
        self.dt_uint16 = h5py.vlen_dtype(np.dtype('uint16'))
        self.dt_uint32 = h5py.vlen_dtype(np.dtype('uint32'))
        self.dt_bytes = h5py.vlen_dtype(np.dtype('uint8'))
        self.dt_string = h5py.string_dtype(encoding='utf-8')
    
    def save_sample_to_shard(self, sample_name, features):
        try:
            if self.samples_in_current_shard >= self.shard_size:
                self._get_new_shard()
            
            grp = self.h5_file.create_group(sample_name)
            grp.create_dataset(
                'features', 
                data=features.astype(np.float32), 
                compression="gzip", 
                compression_opts=4 
            )
            
            self.samples_in_current_shard += 1
        except Exception as e:
            print(e)
            

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@PREEXTRACTORS.register_module()
class IMG_Extractor(ExtractorBase):

    @torch.no_grad()
    def extract(self):
        self.model.eval()

        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Feature Pre-extracting >>>>>>>>>>>>>>>>>")
        self.logger.info(f"Rank {self.rank}: Starting extraction...")
        
        for i, input_dict in enumerate(self.data_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
                features = self.model(input_dict['images'])
            
            features = features.cpu().numpy()
            names = input_dict["name"]
            
            bs = 0
            for j, be in enumerate(input_dict['image_offset']):
                self.save_sample_to_shard(names[j], features[bs:be])
                bs = be
            
            if i % 10 == 0 and self.rank == 0:
                self.logger.info(f"Progress: {i}/{len(self.data_loader)} batches")

        if self.h5_file:
            print("Closing h5 file")
            self.h5_file.close()

        comm.synchronize()
        self.logger.info(f"Rank {self.rank}: Extraction finished.")

    @staticmethod
    def collate_fn(batch):
        return batch

