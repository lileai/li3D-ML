"""
Misc Losses

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES
from .circle_loss import WeightedCircleLoss
from ..utils.misc import pairwise_distance, isotropic_transform_error
from ..utils.registration.procrustes import apply_transform


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            size_average=None,
            reduce=None,
            reduction="mean",
            label_smoothing=0.0,
            loss_weight=1.0,
            ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
            self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
                F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                        torch.sum(
                            pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                        )
                        + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class CoarseMatchingLoss(nn.Module):
    def __init__(self,
                 positive_margin=0.1,
                 negative_margin=1.4,
                 positive_optimal=0.1,
                 negative_optimal=1.4,
                 log_scale=24,
                 positive_overlap=0.1,
                 weight_coarse_loss=1.0):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            positive_margin,
            negative_margin,
            positive_optimal,
            negative_optimal,
            log_scale,
        )
        self.positive_overlap = positive_overlap
        self.weight_coarse_loss = weight_coarse_loss

    def forward(self, pred, target):
        ref_feats = pred['ref_feats_c']
        src_feats = pred['src_feats_c']
        gt_node_corr_indices = target['gt_node_corr_indices']
        gt_node_corr_overlaps = target['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return self.weight_coarse_loss * loss


@LOSSES.register_module()
class FineMatchingLoss(nn.Module):
    def __init__(self, positive_radius=0.05,
             weight_fine_loss=1.0):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = positive_radius
        self.weight_fine_loss = weight_fine_loss

    def forward(self, pred, target):
        ref_node_corr_knn_points = pred['ref_node_corr_knn_points']
        src_node_corr_knn_points = pred['src_node_corr_knn_points']
        ref_node_corr_knn_masks = pred['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = pred['src_node_corr_knn_masks']
        matching_scores = pred['matching_scores']
        transform = target['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return self.weight_fine_loss * loss

# if __name__ == '__main__':
#     # ---- 随机生成一个 batch ----
#     B = 4
#     t_pred = torch.randn(B, 3)  # 预测平移
#     q_pred = torch.nn.functional.normalize(
#         torch.randn(B, 4), dim=1)  # 预测四元数（已归一）
#     pred = {'t': t_pred, 'q': q_pred}
#
#     # 真值 4×4 齐次矩阵
#     target = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)  # [B,4,4]
#     # 让旋转和平移稍微偏离单位矩阵
#     with torch.no_grad():
#         target[:, :3, 3] = torch.randn(B, 3) * 0.1  # 小位移
#         target[:, :3, :3] += torch.randn(B, 3, 3) * 0.05  # 小旋转扰动
#
#     # ---- 计算 loss ----
#     criterion = PoseLoss(w_trans=1.0, w_rot=1.0, reduction='mean')
#     loss = criterion(pred, target)
#     print('PoseLoss =', loss.item())
#
# # ---------------- 1. 构造 3 个样本 ----------------
# # 样本 0: 5 个点
# src0 = torch.randn(5, 3)
# tgt0 = torch.randn(7, 3)
# R0 = torch.randn(3, 3)
# t0 = torch.randn(3)
#
# # 样本 1: 4 个点
# src1 = torch.randn(4, 3)
# tgt1 = torch.randn(6, 3)
# R1 = torch.randn(3, 3)
# t1 = torch.randn(3)
#
# # 样本 2: 6 个点
# src2 = torch.randn(6, 3)
# tgt2 = torch.randn(9, 3)
# R2 = torch.randn(3, 3)
# t2 = torch.randn(3)
#
# # ---------------- 2. 拼成一条 & 构造 offset ----------------
# pred = {
#     'coord': torch.cat([src0, src1, src2], dim=0),          # [15,3]
#     'coord_transformed': torch.cat([tgt0, tgt1, tgt2], dim=0),  # [22,3]
#     'R': torch.stack([R0, R1, R2], dim=0),                  # [3,3,3]
#     't': torch.stack([t0, t1, t2], dim=0),                  # [3,3]
#     'offset': torch.tensor([5, 9, 15], dtype=torch.long),   # [B]   右端点
#     'offset_trans': torch.tensor([7, 13, 22], dtype=torch.long)  # [B]
# }
#
# # ---------------- 3. 计算 MSE ----------------
# criterion = ReconstructionMSE(loss_weight=1.0, reduction='mean')
# loss = criterion(pred, None)
# print('ReconstructionMSE =', loss.item())
