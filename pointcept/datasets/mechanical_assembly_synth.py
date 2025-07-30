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
from PIL import Image
import torchvision.transforms as transforms
from pytorch3d.renderer import (
    PerspectiveCameras,
)


@DATASETS.register_module("MechanicalAssemblySynth")
class MechanicalAssemblySynth(Dataset):

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
        recompute_clustering=False,
        use_clustering="random",
        image_transform=None
    ):
        super(MechanicalAssemblySynth, self).__init__()
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

        if recompute_clustering:
            logger.info("Recomputing clustering for all data...")
            for dir in self.data_list:
                if os.path.exists(f'{self.data_root}/{dir}/0_grp.json'):
                    continue

                print(f"Processing {dir}...")
                self.prepare_clustering(dir)
            logger.info("Clustering recomputed.")
        
        self.use_clustering = use_clustering


        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # or 518 for ViT-Giant
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def prepare_clustering(self, dir, annotations=None):

        mesh = trimesh.load(f'{self.data_root}/{dir}/visible1.ply')
        vertices = torch.from_numpy(mesh.vertices.astype(np.float32))
        faces = torch.from_numpy(mesh.faces.astype(np.int64))
        ind1 = segment_mesh(vertices, faces, 0.0001, 5).numpy()
        ind2 = segment_mesh(vertices, faces, 0.001, 10).numpy()
        ind3 = segment_mesh(vertices, faces, 0.001, len(vertices) // 5).numpy()

        frames = sorted(glob.glob(os.path.join(self.data_root, dir, '*.ply')))
        annotations = sorted(glob.glob(os.path.join(self.data_root, dir, '*.json')))
        annotations = [a for a in annotations if not a.endswith('grp.json')]

        knn = NearestNeighbors(n_neighbors=1)        
        knn.fit(mesh.vertices) 

        groupings = {}

        for i, (annotation, frame) in enumerate(zip(annotations, frames)):    
            print(frame)       
            point_cloud = trimesh.load(frame).vertices
            distances, indices = knn.kneighbors(point_cloud)
                        
            groupings['seg_indices1'] = ind1[indices.flatten()].tolist()
            groupings['seg_indices2'] = ind2[indices.flatten()].tolist()
            groupings['seg_indices3'] = ind3[indices.flatten()].tolist()

            with open(annotation.replace('.json', '_grp.json'), 'w') as json_file:
                json.dump(groupings, json_file)
        
            with open(annotation) as f:
                annotations = json.load(f)
            
            if 'seg_indices' in annotations:
                annotations.pop('seg_indices')
            if 'seg_indices2' in annotations:
                annotations.pop('seg_indices2')
            
            with open(annotation, 'w') as f:
                json.dump(annotations, f)

        cached_path = os.path.join(self.data_root, dir, 'cached.pth')
        if os.path.exists(cached_path):
            os.remove(cached_path)

    def update_semantic_labels(self, raw_data):
        for file in self.data_list:
            print(f"Processing {file}...")
            with open(os.path.join(raw_data, file, 'meta.json'), 'r') as f:
                meta = json.load(f)

            classes = {}
            for part, cls  in meta.items():
                if cls not in classes:
                    classes[cls] = len(classes)

            annotations = sorted(glob.glob(os.path.join(self.data_root, file, '*.json')))
            annotations = [a for a in annotations if not a.endswith('grp.json')]

            for annotation in annotations:
                with open(annotation, 'r') as f:
                    data = json.load(f)

                data['classes'] = classes

                semantic = np.asarray(data['semantic_id'])
                instance = np.asarray(data['instance_id'])

                for i, part in enumerate(data['part_files']):
                    semantic[instance == i] = classes[meta[part]]
                
                data['semantic_id'] = semantic.tolist()

                with open(annotation, 'w') as f:
                    json.dump(data, f)

    def get_clustering(self) -> str:
        if self.use_clustering == "random":
            if np.random.rand() < 0.33:
                return 'seg_indices1'
            elif np.random.rand() < 0.66:
                return 'seg_indices2'
            else:
                return 'seg_indices3'
        elif self.use_clustering == "seg_indices1":
            return 'seg_indices1'
        elif self.use_clustering == "seg_indices2":
            return 'seg_indices2'
        elif self.use_clustering == "seg_indices3":
            return 'seg_indices3'
        else:
            raise NotImplementedError(f"Unknown clustering method: {self.use_clustering}")
            
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

    def pixel_point_matches(
                self,
                pts_world: np.ndarray,
                P: np.ndarray,                 # 3×4 projection matrix
                img_size: tuple[int, int],     # (H, W)
    ):
            """
            Project a point cloud, keep one front‑most point per pixel,
            and return pixel <‑‑> point index correspondences.

            Returns
            -------
            pix_uv  : (M,2) int   integer (u, v) pixel coords
            pc_idx  : (M,)  int   indices into `pts_world`
            """
            
            H, W = img_size
            N = len(pts_world)

            # Homogeneous coordinates
            pts_h = np.c_[pts_world, np.ones(N)]         # (N, 4)
            img_h = pts_h @ P.T                          # (N, 3)

            x_proj = img_h[:, 0]
            y_proj = img_h[:, 1]
            z_proj = img_h[:, 2]

            # Valid depth mask
            valid = z_proj > 1e-6
            if not np.any(valid):
                return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

            x = x_proj[valid] / z_proj[valid]
            y = y_proj[valid] / z_proj[valid]
            z = z_proj[valid]
            pts_idx = np.nonzero(valid)[0]

            # Convert to pixel coordinates
            u = ((1 - x) * W / 2).round().astype(int)
            v = ((1 - y) * H / 2).round().astype(int)

            # Keep only points inside image
            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, z, pts_idx = u[in_bounds], v[in_bounds], z[in_bounds], pts_idx[in_bounds]

            # Initialize depth buffer and index buffer
            depth_buffer = np.full((H, W), np.inf)
            index_buffer = np.full((H, W), -1, dtype=int)

            for i in range(len(u)):
                ui, vi = u[i], v[i]
                if z[i] < depth_buffer[vi, ui]:
                    depth_buffer[vi, ui] = z[i]
                    index_buffer[vi, ui] = pts_idx[i]

            # Extract valid pixels and corresponding point indices
            valid_mask = index_buffer >= 0
            v_coords, u_coords = np.nonzero(valid_mask)
            pix_uv = np.stack([u_coords, v_coords], axis=1)
            pc_idx = index_buffer[v_coords, u_coords]

            return pix_uv, pc_idx

    def get_data(self, idx):

        idx = idx % len(self.data_list)
        dir = self.data_list[idx]

        annotations = sorted(glob.glob(os.path.join(self.data_root, dir, '*.json')))
        annotations = [a for a in annotations if not a.endswith('grp.json')]

        if self.cache and os.path.exists(os.path.join(self.data_root, dir, 'cached.pth')):
            try:
                data=torch.load(os.path.join(self.data_root, dir, 'cached.pth'))
                data['seg_indices'] = data[self.get_clustering()]
                data.pop('seg_indices1')
                data.pop('seg_indices2')
                data.pop('seg_indices3')
                return data
            except Exception as e:
                print(f"Error loading {dir}: {e}")
                os.remove(os.path.join(self.data_root, dir, 'cached.pth'))
        
        vertices = []
        semantic_id = []
        instance_id = []
        frame_id = []
        seg_indices1 = []
        seg_indices2 = []
        seg_indices3 = []

        # try:
        for i, annotation in enumerate(annotations):
            groups = annotation.replace('.json', '_grp.json')
            frame = annotation.replace('.json', '.ply')
            labels = json.load(open(annotation))
            groupings = json.load(open(groups))

            instance_id.append(labels['instance_id'])
            if 'semantic_id' in labels:
                semantic_id.append(np.asarray(labels['semantic_id']))
            else:
                semantic_id.append(np.asarray([0] * len(labels['instance_id'])))

            semantic_mapping = np.asarray([self.class_mapping[c] for c in labels['classes']])
            if 'seg_indices' in labels:
                semantic_id[-1] = semantic_mapping[semantic_id[-1]]

            frame_id.append(np.asarray([i] * len(labels['instance_id'])))     
            seg_indices1.append(np.asarray(groupings['seg_indices1']))
            seg_indices2.append(np.asarray(groupings['seg_indices2']))     
            seg_indices3.append(np.asarray(groupings['seg_indices3']))     

            pcd = trimesh.load(frame)
            vertices.append(pcd.vertices.astype(np.float32))  

        vertices = np.concatenate(vertices, axis=0)
        instance_id = np.concatenate(instance_id, axis=0)
        semantic_id = np.concatenate(semantic_id, axis=0)
        frame_id = np.concatenate(frame_id, axis=0)
        seg_indices1 = np.concatenate(seg_indices1, axis=0)
        seg_indices2 = np.concatenate(seg_indices2, axis=0)
        seg_indices3 = np.concatenate(seg_indices3, axis=0)
        # except Exception as e:
        #     print(f"Error loading {dir}: {e}")
        #     return self.get_data(idx + 1)

        # Estimate normals for the point cloud
        mesh = trimesh.Trimesh(vertices=vertices, process=False)
        normals = mesh.vertex_normals.copy()
        
        if len(mesh.vertices) < 2048:
            del mesh
            return self.get_data(idx + 1)
        
        keep_ids = np.arange(len(vertices))

        while len(keep_ids) > 400000:
            keep_ids = keep_ids[::2]
            vertices = vertices[::2]
    
        images, mappings_src, mappings_tgt = [], [], []

        if os.path.exists(os.path.join(self.data_root, dir, 'color')): 
            image_paths = sorted(glob.glob(os.path.join(self.data_root, dir,  'color/*.png')))[::1]
            P_paths = sorted(glob.glob(os.path.join(self.data_root, dir,  'poses/*P.txt')))[::1]

            for i, (image_path, P_path) in enumerate(zip(image_paths, P_paths)):
                image = Image.open(image_path).convert('RGB')
                images.append(image)

                P = np.loadtxt(P_path)

                src, tgt = self.pixel_point_matches(
                    pts_world=vertices,
                    P=P,
                    img_size=(224, 224)
                )

                src = np.stack([np.ones(len(src)) * i, src[:, 0], src[:, 1]], axis=1)  # (N, 3)

                mappings_src.append(src.astype(np.int32))
                mappings_tgt.append(tgt.astype(np.int32))
            
            mappings_src = np.concatenate(mappings_src, axis=0)
            mappings_tgt = np.concatenate(mappings_tgt, axis=0)

        while np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)) < 80:
            vertices *= 2

        while np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)) > 400:
            vertices /= 2
            # keep_ids = keep_ids[::2]
            # vertices = vertices[::2]

        try:
            data = {
                'coord': vertices,
                'normal': normals[keep_ids],
                'instance': instance_id[keep_ids],
                'segment': semantic_id[keep_ids],
                'id': idx,
                'path': self.data_list[idx],
                'frame_id': frame_id,
                'name': self.get_data_name(idx),
                'seg_indices1': seg_indices1[keep_ids],
                'seg_indices2': seg_indices2[keep_ids],
                'seg_indices3': seg_indices3[keep_ids],
                'mappings_src': mappings_src,
                'mappings_tgt': mappings_tgt,
                'images': images,
            }

            if self.cache:
                torch.save(data, os.path.join(self.data_root, dir, 'cached.pth'))

        except Exception as e:
            print(f"Error processing {dir}: {e}")
            return self.get_data(idx + 1)

        data['seg_indices'] = np.asarray(data[self.get_clustering()])
        data.pop('seg_indices1')
        data.pop('seg_indices2')
        data.pop('seg_indices3')

    
        
        return data

    def get_data_name(self, idx):
        data_name = self.data_list[idx]
        data_name = data_name.replace(".ply", "")
        data_name = data_name.replace(".obj", "")
        data_name = data_name.replace(".stl", "")
        data_name = data_name.replace("/", "_")
        return data_name

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)  

        if self.image_transform is not None:
            if 'images' in data_dict:
                data_dict['images'] = np.stack([
                    self.image_transform(image).numpy() for image in data_dict['images']
                ])
            else:
                data_dict['images'] = []

        data_dict = self.transform(data_dict)

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)

        if self.image_transform is not None:
            if 'images' in data_dict:
                data_dict['images'] = np.stack([
                    self.image_transform(image).numpy() for image in data_dict['images']
                ])
            else:
                data_dict['images'] = []

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

