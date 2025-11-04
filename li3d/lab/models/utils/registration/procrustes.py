import torch
import torch.nn as nn
from ..misc import pairwise_distance


@torch.no_grad()
def get_node_correspondences(
    ref_nodes,
    src_nodes,
    ref_knn_points,
    src_knn_points,
    transform,
    pos_radius,
    ref_masks=None,
    src_masks=None,
    ref_knn_masks=None,
    src_knn_masks=None,
):
    r"""Generate ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.

    Args:
        ref_nodes: torch.Tensor (M, 3)
        src_nodes: torch.Tensor (N, 3)
        ref_knn_points: torch.Tensor (M, K, 3)
        src_knn_points: torch.Tensor (N, K, 3)
        transform: torch.Tensor (4, 4)
        pos_radius: float
        ref_masks (optional): torch.BoolTensor (M,) (default: None)
        src_masks (optional): torch.BoolTensor (N,) (default: None)
        ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
        src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)

    Returns:
        corr_indices: torch.LongTensor (C, 2)
        corr_overlaps: torch.Tensor (C,)
    """
    src_nodes = apply_transform(src_nodes, transform)
    src_knn_points = apply_transform(src_knn_points, transform)

    # generate masks
    if ref_masks is None:
        ref_masks = torch.ones(size=(ref_nodes.shape[0],), dtype=torch.bool).cuda()
    if src_masks is None:
        src_masks = torch.ones(size=(src_nodes.shape[0],), dtype=torch.bool).cuda()
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones(size=(ref_knn_points.shape[0], ref_knn_points.shape[1]), dtype=torch.bool).cuda()
    if src_knn_masks is None:
        src_knn_masks = torch.ones(size=(src_knn_points.shape[0], src_knn_points.shape[1]), dtype=torch.bool).cuda()

    node_mask_mat = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))  # (M, N)

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1), dim=-1)  # (M, K)
    ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
    ref_max_dists = ref_knn_dists.max(1)[0]  # (M,)
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (N, K)
    src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
    src_max_dists = src_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(pairwise_distance(ref_nodes, src_nodes))  # (M, N)
    intersect_mat = torch.gt(ref_max_dists.unsqueeze(1) + src_max_dists.unsqueeze(0) + pos_radius - dist_mat, 0)
    intersect_mat = torch.logical_and(intersect_mat, node_mask_mat)
    sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat, as_tuple=True)

    # select potential patch pairs
    ref_knn_masks = ref_knn_masks[sel_ref_indices]  # (B, K)
    src_knn_masks = src_knn_masks[sel_src_indices]  # (B, K)
    ref_knn_points = ref_knn_points[sel_ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[sel_src_indices]  # (B, K, 3)

    point_mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))  # (B, K, K)

    # compute overlaps
    dist_mat = pairwise_distance(ref_knn_points, src_knn_points)  # (B, K, K)
    dist_mat.masked_fill_(~point_mask_mat, 1e12)
    point_overlap_mat = torch.lt(dist_mat, pos_radius ** 2)  # (B, K, K)
    ref_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1), dim=-1).float()  # (B,)
    src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2), dim=-1).float()  # (B,)
    ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()  # (B,)
    src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    overlap_masks = torch.gt(overlaps, 0)
    ref_corr_indices = sel_ref_indices[overlap_masks]
    src_corr_indices = sel_src_indices[overlap_masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    corr_overlaps = overlaps[overlap_masks]

    return corr_indices, corr_overlaps

def apply_transform(points, transform, normals=None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points

def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H.cpu())  # H = USV^T
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.0, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights=None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
        )
