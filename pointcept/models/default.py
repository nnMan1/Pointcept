import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model
from pointcept.utils.visualization import to_o3d, colors
import open3d as o3d


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])

            if loss is None:
                pass

            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class GroupingSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None, final_in_channels = 128, num_classes = 4):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.final = nn.Linear(final_in_channels, num_classes)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]

        bb_features = self.backbone(input_dict)

        seg_logits_gr = []
        seg_logits_pt = []


        tgts_gr = []

        bb = 0
        for be in input_dict['offset']:
            groups = input_dict['seg_indices'][bb:be]
            group_size = torch.bincount(groups)

            if 'segment' in input_dict.keys():
                labels = input_dict['segment'][bb:be]

            fts = bb_features[bb:be]

            labels = labels[group_size[groups] > 50]
            coords = input_dict['coord'][bb:be][group_size[groups] > 50]
            fts = fts[group_size[groups] > 50]
            groups = groups[group_size[groups] > 50]


            filter = torch_scatter.scatter_min(labels, groups)[0] != torch_scatter.scatter_max(labels, groups)[0]
            filter = filter.cpu()
            print(group_size[filter])

            # if self.training:
            #     filter = group_size[groups] > 50
            #     fts = fts[filter]
            #     labels = labels[filter] 
            #     groups = groups[filter]

            group_fts = torch_scatter.scatter_mean(fts, groups, dim=0)

            if 'segment' in input_dict.keys():
                if any(torch_scatter.scatter_min(labels, groups)[0] != torch_scatter.scatter_max(labels, groups)[0]):
                    # ids = torch.where(torch_scatter.scatter_min(labels, groups)[0] == torch_scatter.scatter_max(labels, groups)[0])
                    # print(ids, filter.sum())
                    pcd = to_o3d(input_dict['coord'][bb:be][filter[groups.cpu()]].cpu(), verts_colors=colors[groups.cpu()[filter[groups.cpu()]] % len(colors)])
                    o3d.io.write_point_cloud('pcd.ply', pcd)
                    raise Exception("Wrong annotations")

                tgts_gr.append(torch_scatter.scatter_min(labels, groups)[0])

            sl = self.final(group_fts)
            seg_logits_gr.append(sl)

            seg_logits_pt.append(sl[groups])
                
            bb = be
        
        seg_logits_pt = torch.cat(seg_logits_pt, 0)
        seg_logits_gr = torch.cat(seg_logits_gr, 0)

        if 'segment' in input_dict.keys():
            tgts_gr = torch.cat(tgts_gr)

        # train
        if self.training:
            loss = self.criteria(seg_logits_gr, tgts_gr)
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits_pt, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits_pt)
        # test
        else:
            return dict(seg_logits=seg_logits_pt)

@MODELS.register_module()
class GroupingSegmentorV2(nn.Module):
    def __init__(self, backbone=None, criteria=None, final_in_channels = 128, num_classes = 4):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.final = nn.Linear(final_in_channels, num_classes)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]

        bb_features = self.backbone(input_dict)

        seg_logits_gr = []
        seg_logits_pt = []


        tgts_gr = []

        bb = 0
        for be in input_dict['offset']:
            groups = input_dict['seg_indices'][bb:be]
            fts = bb_features[bb:be]

            group_fts = torch_scatter.scatter_mean(fts, groups, dim=0)

            if 'segment' in input_dict.keys():
                tgts_gr.append(torch_scatter.scatter_min(input_dict['segment'][bb:be], groups)[0])

            sl = self.final(group_fts)
            seg_logits_gr.append(sl)

            seg_logits_pt.append(sl[groups])
                
            bb = be
        
        seg_logits_pt = torch.cat(seg_logits_pt, 0)
        seg_logits_gr = torch.cat(seg_logits_gr, 0)
        tgts_gr = torch.cat(tgts_gr)

        # print(seg_logits_pt.shape, seg_logits_gr.shape, tgts_gr.max(),tgts_gr.min(), tgts_gr.dtype)
        # exit(0)

        # train
        if self.training:
            loss = self.criteria(seg_logits_gr, tgts_gr)
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits_pt, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits_pt)
        # test
        else:
            return dict(seg_logits=seg_logits_pt)

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
