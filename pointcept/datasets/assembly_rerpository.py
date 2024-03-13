import sys
import json
import os
import glob
import os.path as osp
import shutil
from typing import Callable, List, Optional, Union

import h5py
import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes, IO
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points

from scipy.spatial import ConvexHull
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class Assembly(InMemoryDataset):
   
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        augment: Optional[Callable] = None,
    ):

        self.augment = augment
        self.split = split
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))
            
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        with open(osp.join(self.raw_dir, f'{self.split}_files.txt')) as f:
            filenames = f.readlines()
            filenames = [f.strip() for f in filenames]
                
        return filenames

    @property
    def processed_file_names(self) -> str:
        return [f'{self.split}.pt']
    
    def __get_aligned_bbox(self, meshes):
        
        pointclouds = sample_points_from_meshes(meshes)
        
        centers = pointclouds.mean(axis = 1, keepdim=True)
        
        pointclouds = pointclouds - centers
        
        C = pointclouds.permute(0, 2, 1).matmul(pointclouds)
        e, v = torch.linalg.eigh(C)
        
        rotated_pointcloud = pointclouds.matmul(v.float())
        
        lenghts = rotated_pointcloud.max(axis=1)[0] * 2
        
        return centers[:, 0], v.float(), lenghts       
        
    def process(self):

        data_list = [] 

        for i, sample in enumerate(self.raw_file_names):
            paths = glob.glob(osp.join(self.raw_dir, sample, '*.obj'))
            parts = load_objs_as_meshes(paths)
            t, r, lenghts = self.__get_aligned_bbox(parts)
                                
            if self.augment:
                scans = self.augment(parts)
                
                for scan in scans:
                    scan.assembly = sample
                
                for scan in scans:
                    scan.t = t
                    scan.r = r
                    scan.lenghts = lenghts
                    
                data_list += scans
                
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(self.processed_paths, j)
        print(len(data_list))
        torch.save(self.collate(data_list), self.processed_paths[j])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'categories={self.categories})')


if __name__ == '__main__':
    
    import sys
    sys.path.append('/home')
    
    import open3d as o3d
    
    from utils.preprocessing import MeshScanner
    from utils.visualization import colors, draw_bbox
   
    ds = Assembly('/home/data/AssemblyRepository', augment=MeshScanner(device='cuda'))
    print(len(ds))
    for d in ds[5::6]:
        # if d.assembly != '0/stl3/00001648':
        #     continue
        
        pos = d.pos
        geom = o3d.geometry.PointCloud()
        geom.points = o3d.utility.Vector3dVector(pos.cpu().numpy())
        geom.colors = o3d.utility.Vector3dVector(colors[d.y % len(colors)])
        
        geoms = [geom]
        
        # for r, t, lwh in zip(d.r, d.t, d.lenghts):
        #     geoms.append(draw_bbox(r, t, lwh))
        
        o3d.visualization.draw_geometries(geoms, window_name=d.assembly)
        
    
