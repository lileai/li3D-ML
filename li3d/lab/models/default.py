import torch
import torch.nn as nn

from .losses import build_criteria
from .utils.structure import Point
from ..utils.misc import matrix_to_quat
from .builder import MODELS, build_model
from .utils.registration.procrustes import get_node_correspondences


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
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict

@MODELS.register_module()
class DefaultRegistration(nn.Module):
    def __init__(
        self,
        backbone=None,
        head=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.head = build_model(head)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict):
        return_dict = {}
        input_dict["transform_matrix"] = input_dict["transform_matrix"].view(-1, 4, 4)
        source_dict = {
            "feat": input_dict["orig"],
            "coord": input_dict["coord"],
            "grid_coord": input_dict["grid_coord"],
            "offset": input_dict["offset"],
        }
        target_dict = {
            "feat": input_dict["trans"],
            "coord": input_dict["coord_transformed"],
            "grid_coord": input_dict["grid_coord_transformed"],
            "offset": input_dict["offset_trans"],
        }
        source_dict = Point(source_dict)
        target_dict = Point(target_dict)
        source_dict.sparsify()
        target_dict.sparsify()

        source_list = self.backbone(source_dict.sparse_conv_feat)
        target_list = self.backbone(target_dict.sparse_conv_feat)
        source_dict, target_dict, corr_scores, estimated_transform = self.head(source_dict, target_dict,source_list, target_list)
        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            target_dict.sp_coord,
            source_dict.sp_coord,
            target_dict.corr_knn_coord,
            source_dict.corr_knn_coord,
            input_dict["transform_matrix"],
            self.matching_radius,
            ref_masks=target_dict.sp_mask,
            src_masks=source_dict.sp_mask,
            ref_knn_masks=target_dict.corr_knn_mask,
            src_knn_masks=source_dict.corr_knn_mask,
        )
        pred = {
            "src_points": input_dict["coord"],
            "ref_points_c": target_dict.sp_coord,
            "src_points_c": source_dict.sp_coord,
            "ref_feats_c": target_dict.sp_feat,
            "src_feats_c": source_dict.sp_feat,
            "ref_corr_points": target_dict.corr_knn_point,
            "src_corr_points": source_dict.corr_knn_point,
            "ref_node_corr_indices": target_dict.corr_knn_indices,
            "src_node_corr_indices": source_dict.corr_knn_indices,
            "ref_node_corr_knn_points": target_dict.corr_knn_coord,
            "src_node_corr_knn_points": source_dict.corr_knn_coord,
            "ref_node_corr_knn_masks": target_dict.corr_knn_mask,
            "src_node_corr_knn_masks": source_dict.corr_knn_mask,
            "matching_scores": corr_scores,
            "estimated_transform": estimated_transform,
        }
        target = {
            "gt_node_corr_indices": gt_node_corr_indices,
            "gt_node_corr_overlaps": gt_node_corr_overlaps,
            "transform": input_dict["transform_matrix"],
        }
        # train
        if self.training:
            loss = self.criteria(pred, target)
            return_dict["loss"] = loss
        # eval
        elif "transform_matrix" in input_dict.keys():
            loss = self.criteria(pred, target)
            return_dict["loss"] = loss
            return_dict["pred"] = pred
            return_dict["target"] = target_dict
        # test
        else:
            return_dict["pred"] = pred
        return return_dict
