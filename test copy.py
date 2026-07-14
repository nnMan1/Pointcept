from pointcept.datasets import build_dataset
from pointcept.models import build_model
from torch import Tensor
from pointcept.utils.visualization import pca_features_visualization
import torch_scatter
import torch
from torch import nn
from torch_geometric.nn import fps, knn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms


classes={"other": 0,  
        "gear": 0, 
        "nut": 0, 
        "screw": 0, 
        "axe": 0}

class_names = ["other"]

segment_ignore_index = ( -1, )

# dataset = build_dataset(dict(
#         type='PartNetInstance',
#         split='train',
#         data_root='data/Partnet/data/partnet/ins_seg_h5',
#         categories=['Chair'],
#         transform=[
#             dict(type='PointCloudMultiView', num_views=20, image_size=(512, 512), point_size=5),
#             dict(type="Copy", keys_dict={ "images": "images_origin" }),    
#             dict(type='Permute', key='images', permutation=[0, 3, 1, 2]),
#             dict(type='ImgResize', target_size=(448, 448), keys=['images']),
#             dict(type='ImgNormalize', key='images', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             dict(type="CenterShift", apply_z=True),  
#             dict(
#                 type="Copy",
#                 keys_dict={
#                     "coord": "origin_coord",
#                     # "segment": "origin_segment",
#                     "instance": "origin_instance"
#                 },
#             ),        
#             # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
#             dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
#             dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.5),
#             dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.5),
#             dict(type="RandomScale", scale=[0.9, 1.1]),
#             dict(type="RandomFlip", p=0.8),
#             dict(type="RandomJitter", sigma=0.001, clip=0.02),
#             dict(
#                 type="GridSample",
#                 grid_size=0.01,
#                 hash_type="fnv",
#                 mode="train",
#                 return_inverse=True,
#                 return_grid_coord=True,
#                 keys=("coord", "normal", "instance", "color"),
#             ),
#             dict(
#                 type="InstanceParser",
#                 segment_ignore_index=segment_ignore_index,
#                 instance_ignore_index=-1,
#             ),
#             dict(type="ToTensor"),
#             dict(
#                 type="Collect",
#                 keys=(
#                     "coord",
#                     "color",
#                     "instance",
#                     "images",
#                     "mappings_src",
#                     "mappings_tgt",
#                     "instance_segment",
#                     "inverse",
#                     "images_origin"
#                 ),
#                 feat_keys=("coord"),
#                 offset_keys_dict=dict(
#                     offset="coord", 
#                     origin_offset="origin_coord", 
#                     image_offset="images", 
#                     mappings_offset="mappings_src",
#                     instance_segment_offset="instance_segment"
#                 ),
#             ),
#         ]
#     ),
# )

dataset = build_dataset(dict(
    type='HDF5_Dataset',
    split='test',
    data_root=
    '/home/data/segment-assembly-synthetic/data/partnet/processed/test/',
    transform=[
        dict(type='CenterShift', apply_z=True),
        dict(
            type='Copy',
            keys_dict=dict(
                coord='origin_coord',
                segment='origin_segment',
                instance='origin_instance')),
        dict(
            type='GridSample',
            grid_size=1,
            hash_type='fnv',
            mode='train',
            return_grid_coord=True,
            return_inverse=True,
            keys=('coord', 'segment', 'instance')),
        dict(type='CenterShift', apply_z=False),
        dict(
            type='InstanceParser',
            segment_ignore_index=(-1, ),
            instance_ignore_index=-1),
        dict(type='FPSSeed', n_points=100),
        dict(type='ToTensor'),
        dict(
            type='Collect',
            keys=('coord', 'grid_coord', 'segment', 'instance',
                    'instance_segment', 'images', 'mappings_src',
                    'mappings_tgt', 'origin_coord', 'origin_segment',
                    'origin_instance', 'path', 'name',
                    'inverse', 'instance_segment'),
            feat_keys='coord',
            offset_keys_dict=dict(
                offset='coord',
                origin_offset='origin_coord',
                image_offset='images',
                mappings_offset='mappings_src',
                instance_segment_offset='instance_segment'))
    ],
    test_mode=False,
    classes=dict(other=0, gear=0, nut=0, screw=0, axe=0)))

model = build_model(dict(
        type='Image2PointCLoud',
        model_type="DinoV2",
        model_name='facebook/dinov2-large',
        fts_dim=1024,
        merge_strategy='mean',
        return_features=['feat'],
        freeze_backbone=True,
        freeze_backbone_bn=False,
        project_fts=False
))

image_transform = transforms.Compose([
            transforms.Resize((448, 448)),  # or 518 for ViT-Giant
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

model.to('cuda')

from pointcept.utils.visualizer import *
visualizer = PointCloudFeatureVisualizer(save_path="results", extension="ply", n_components=3)
img_visualizer = ImageVisualizer(save_path="results", extension='png')
# visualizer = PointCloudVisuzlizer(save_path="results", extension="ply")
from sklearn.cluster import KMeans

def spectral_clustering_cluster_qr(features, k, device='cuda'):
    # features: (N, C)
    # k: number of clusters
    # ratio: ratio of eigenvectors to use for clustering
    # device: cpu or cuda
    # returns: (N, 1) cluster assignments
    #
    
    # compute the impact of each dimension on the clustering
    # (N, C)
    features = features.to(device)
    features = features - features.mean(dim=0, keepdim=True)
    
    # (C, C)
    print(features.shape)
    cov = features.T @ features
    
    # (C, C)
    _, S, _ = torch.svd(cov)
    
    # (C, 1)
    S = S.unsqueeze(-1)
    
    # (C, 1)
    S = torch.sqrt(S)
    
    # (C, 1)
    S = torch.reciprocal(S)
    
    # (C, C)
    S = torch.diag(S.squeeze())
    
    # (C, C)
    cov_inv = S @ cov @ S
    
    # get the best k dimensions
    # (C, k)
    _, _, V = torch.svd(cov_inv)
    
    # (C, k)
    V = V[:, :k]
    
    # (N, k)
    features = features @ V

    # get the cluster assignments
    # (N)
    return KMeans(n_clusters=k, random_state=0, n_init="auto").fit(features.cpu())

class GeometricAwareFeatureAggregation(nn.Module):
    def __init__(
            self,
            levels=torch.tensor([256, 256]),
            neighborhood=torch.tensor([5, 20]),
            aggregation_type=['xyz', 'sem'],
            weight_by_distance=torch.tensor([False, False]),
            upsample_to_original_size=True
    ):
        super().__init__()

        assert len(levels) == len(neighborhood) == len(aggregation_type) == len(
            weight_by_distance), '`levels`, `neighborhood`, `aggregation_type`, and `weight_by_distance` must be same length'

        self.levels = levels
        self.neighborhood = neighborhood
        self.aggregation_type = aggregation_type
        self.weight_by_distance = weight_by_distance
        self.upsample_to_original_size = upsample_to_original_size

    def forward(self, point_cloud, features, per_point_vectors):
        superpoint_features = features
        points = point_cloud[:, :3]
        labels = per_point_vectors.float()
        for level, neighbors, aggregation_type, weight_by_distance in zip(self.levels, self.neighborhood,
                                                                          self.aggregation_type,
                                                                          self.weight_by_distance):
            # Extract superpoints
            index = fps(points, ratio=level / len(points))
            superpoints = points[index]
            labels = labels[:, index]

            # Aggregate features based either on xyz space or on feature space
            # Also, features might be weighted according to their distance
            # from the superpoint
            if aggregation_type == 'xyz':
                sm = superpoints
                lg = points
            else:
                sm = superpoint_features[index]
                lg = superpoint_features

            if weight_by_distance:
                values, point_to_superpoint_mapping = torch.cdist(sm, lg).topk(neighbors, largest=False, dim=-1)
                weights = F.softmin(values, dim=-1)

                superpoint_features = (weights.unsqueeze(-1) * superpoint_features[point_to_superpoint_mapping]).sum(
                    dim=1)
            else:
                point_to_superpoint_mapping = knn(
                    lg, sm, neighbors
                )[1].view(len(superpoints), -1)
                superpoint_features = superpoint_features[point_to_superpoint_mapping].mean(dim=1)
            points = superpoints

            # Upsample
            if self.upsample_to_original_size:
                index = self.upsample(superpoints, point_cloud[:, :3])
                points = point_cloud[:, :3]
                superpoint_features = superpoint_features[index].mean(dim=1)
                labels = labels[:, index].mean(dim=-1)

        return points, superpoint_features, labels

    def upsample(self, superpoints, points, closest_superpoints=1):
        point_to_superpoint_mapping = knn(
            superpoints, points, closest_superpoints
        )[1].view(len(points), -1)
        return point_to_superpoint_mapping

    def to(self, device):
        self.levels = self.levels.to(device)
        self.neighborhood = self.neighborhood.to(device)
        return self

for i, sample in enumerate(dataset):

    for k, v in sample.items():
        if isinstance(v, Tensor):
            sample[k] = v.cuda()

    print("saving images", i)

    for j in range(20):
        img_visualizer({
            'image': sample['images'][j].permute(1, 2, 0)
        }, f'sample_{i}_{j}')


    #     mask = sample['mappings_src'][:, 0] == j
    #     mappings_tgt = sample['mappings_tgt'][mask].unique()
    #     # visualizer({
    #     #     'coord': sample['coord'][mappings_tgt],
    #     #     'color': sample['color'][mappings_tgt] / 255
    #     #     # 'feat': superpoint_features,
    #     #     # 'label': kmeans.labels_
    #     # }, f'sample_{i}_{j}')

    # sample['images'] = image_transform(sample['images'].float() / 255).cuda()

    # print(sample['image_offset'])
            
    pred = model(sample)
    # fts =  torch_scatter.scatter_mean(pred['feat'], sample['seg_indices'], dim=0)
    # points, superpoint_features, labels = GeometricAwareFeatureAggregation().to('cuda')(sample['coord'], pred['feat'], sample['coord'].T)
    # kmeans = spectral_clustering_cluster_qr(superpoint_features, 8)

    f = pred['feat'][:, 0] != 0
    print("saving sample")
    visualizer({
        'coord': sample['coord'][f],
        'feat': pred['feat'][f].cpu(),
        # 'label': kmeans.labels_
    }, f'sample_{i}')

    # pca_features_visualization(sample['coord'].cpu().numpy(), fts['feat'].cpu().numpy(), file_path=f"features{i}.ply")
    # print(sample['name'], sample['image_features'].shape, fts['feat'].shape, sample['coord'].shape)

    if i > 3:
        exit(0)