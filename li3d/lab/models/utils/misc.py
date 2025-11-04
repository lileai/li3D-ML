"""
General Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import numpy as np
from itertools import chain


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    """
    b_idx = y1.indices[:, 0]       # (N_voxel,)
    offset = batch2offset(b_idx)   # (B+1,)
    # 第 b 帧 voxel 范围 = offset[b] : offset[b+1]
    feat_b = y1.features[offset[b]:offset[b+1]]
    """
    return torch.cumsum(batch.bincount(), dim=0).long()

@torch.no_grad()
def offset2mask(offset: torch.LongTensor):
    """
    offset: (B+1,)  累积长度
    return: (B, L_max)  True=valid
    """
    B = offset.size(0) - 1
    L_max = (offset[1:] - offset[:-1]).max().item()
    mask = torch.zeros(B, L_max, dtype=torch.bool, device=offset.device)
    for b in range(B):
        mask[b, :offset[b+1]-offset[b]] = True
    return mask


@torch.no_grad()
@torch.no_grad()
def packed2batch(x, coord, batch_idx, pad_value=0.0):
    """
    x:      (N, C)     packed 特征
    coord:  (N, 3)     packed 坐标
    batch_idx: (N,)    batch id
    pad_value: 填充值
    return:
        padded:   (B, L_max, C)  补零特征
        coord_pad:(B, L_max, 3)  补零坐标
        mask:     (B, L_max)     True=有效
    """
    offset = torch.cat([torch.zeros(1, dtype=torch.long, device=batch_idx.device),
                        torch.bincount(batch_idx).cumsum(0)], 0)  # (B+1,)
    B = offset.size(0) - 1
    L_max = (offset[1:] - offset[:-1]).max().item()

    # 预分配
    C = x.shape[1]
    padded    = torch.full((B, L_max, C), pad_value, dtype=x.dtype, device=x.device)
    coord_pad = torch.full((B, L_max, 3), 0.0, dtype=coord.dtype, device=coord.device)
    mask      = torch.zeros(B, L_max, dtype=torch.bool, device=x.device)

    # 逐样本填充
    for b in range(B):
        st, ed = offset[b], offset[b+1]
        l = ed - st
        padded[b, :l]    = x[st:ed]
        coord_pad[b, :l] = coord[st:ed]
        mask[b, :l]      = True

    return padded, coord_pad, mask

@torch.no_grad()
def batch2packed(padded, mask):
    """
    padded: (B, L_max, *)  padded 张量
    mask:   (B, L_max)     True=有效
    return:
        packed: (N, *)  去填充后的张量
    """
    packed_list = []
    for b in range(mask.size(0)):
        l = mask[b].sum().item()  # 有效长度
        packed_list.append(padded[b, :l])
    return torch.cat(packed_list, dim=0)  # (N, *)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

@torch.no_grad()
def pairwise_distance(x, y, normalized=False, channel_first=False):
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances

@torch.no_grad()
def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    r"""Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output

def get_rotation_translation_from_transform(transform):
    r"""Decompose transformation matrix into rotation matrix and translation vector.

    Args:
        transform (Tensor): (*, 4, 4)

    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation

def relative_rotation_error(gt_rotations, rotations):
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / np.pi
    return rre

def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte

def isotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (*, 4, 4)
        transforms (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ['mean', 'sum', 'none']

    gt_rotations, gt_translations = get_rotation_translation_from_transform(gt_transforms)
    rotations, translations = get_rotation_translation_from_transform(transforms)

    rre = relative_rotation_error(gt_rotations, rotations)  # (*)
    rte = relative_translation_error(gt_translations, translations)  # (*)

    if reduction == 'mean':
        rre = rre.mean()
        rte = rte.mean()
    elif reduction == 'sum':
        rre = rre.sum()
        rte = rte.sum()

    return rre, rte
