import math
import torch
import torch_scatter
import torch.nn as nn
import spconv.pytorch as spconv
from einops import rearrange
from .builder import MODELS
from .utils.structure import Point
from .utils.misc import offset2bincount
from ..modules.point_modules import PointModule, PointSequential

# ---------- RoPE ----------
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0] * x.shape[1], x.shape[-1])
    shape = [d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ---------- 稀疏 Transformer Block ----------
class SparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.1, patch_size=128):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.patch_size = patch_size
        self.scale = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = torch.nn.Dropout(0.3)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim, bias=False),
            nn.Dropout(dropout),
        )

    @torch.no_grad()
    def get_padding_and_inverse(self, point: dict):
        """
        输入：
            point : dict，必须包含 point.offset（int64 tensor，长度 = batch）
        返回：
            pad        : [num_padded_points]  扁平序列索引 → 原序列索引
            unpad      : [num_orig_points]    原序列索引   → 扁平序列索引
            cu_seqlens : [num_patches+1]      每个 patch 在扁平序列里的起始位置
        """
        # ---------- 0. 缓存 key ----------
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"

        if all(k in point for k in (pad_key, unpad_key, cu_seqlens_key)):
            return point[pad_key], point[unpad_key], point[cu_seqlens_key]

        # ---------- 1. 基础统计 ----------
        offset = point.offset  # [B+1] 前缀和
        bincount = offset2bincount(offset)  # [B]   每样本真实点数
        ps = self.patch_size

        # 需要 pad 到的长度（向上取整到 ps 倍数，但 ≤ps 的样本保持原长）
        bincount_pad = torch.where(
            bincount > ps,
            ((bincount + ps - 1) // ps) * ps,
            bincount,
        )

        # 原序列 / pad 后序列的累积偏移
        _offset = nn.functional.pad(offset, (1, 0))  # [B+1]
        _offset_pad = nn.functional.pad(
            torch.cumsum(bincount_pad, dim=0), (1, 0)
        )  # [B+1]

        # ---------- 2. 初始化三张表 ----------
        pad_flat = torch.arange(_offset_pad[-1], device=offset.device)  # [N_pad]
        unpad_flat = torch.arange(_offset[-1], device=offset.device)  # [N_orig]
        cu_seqlens_list = []

        # ---------- 3. 逐样本处理 ----------
        for i in range(bincount.shape[0]):
            # 3.1 把 unpad 映射到 pad 后序列
            orig_start, orig_end = _offset[i], _offset[i + 1]
            pad_start, pad_end = _offset_pad[i], _offset_pad[i + 1]
            unpad_flat[orig_start:orig_end] += pad_start - orig_start

            # 3.2 循环填充（仅当需要时）
            if bincount[i] != bincount_pad[i]:
                gap_len = (pad_end - pad_start) - bincount[i]  # 缺口长度
                gap_start = pad_start + bincount[i]  # 缺口起始
                donor_start = pad_start + bincount[i] - ps  # 倒数第二 patch 起点
                pad_flat[gap_start: gap_start + gap_len] = \
                    pad_flat[donor_start: donor_start + gap_len]

            # 3.3 把 pad 映射回原序列
            pad_flat[pad_start:pad_end] -= pad_start - orig_start

            # 3.4 记录该样本各 patch 起始
            patches = torch.arange(
                pad_start, pad_end, step=ps,
                dtype=torch.int32, device=offset.device
            )
            cu_seqlens_list.append(patches)

        # ---------- 4. 拼 cu_seqlens ----------
        cu_seqlens = nn.functional.pad(
            torch.cat(cu_seqlens_list), (0, 1), value=_offset_pad[-1]
        )

        # ---------- 5. 缓存并返回 ----------
        point[pad_key] = pad_flat
        point[unpad_key] = unpad_flat
        point[cu_seqlens_key] = cu_seqlens
        # 任何需要“把原始特征先 pad 成固定 patch 倍数”的地方，就用 pad：feats_padded = feats[pad]
        # 任何需要“把 pad 后的结果还原成原始顺序”的地方，就用 unpad：feats_restored = feats_padded_out[unpad]

        return pad_flat, unpad_flat, cu_seqlens

    def forward(self, x, freqs_cis):
        feat = self.norm1(x.feat)                      # [nnz, C]
        _, C = feat.shape
        pad, unpad, _ = self.get_padding_and_inverse(x)
        freqs_cis = freqs_cis[pad]
        qkv = self.qkv(feat[pad]).view(-1, 3, self.patch_size, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(1)                            # [nnz, n_heads, head_dim]
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = rearrange(q, 'n p h d -> n h p d')
        k = rearrange(k, 'n p h d -> n h p d')
        v = rearrange(v, 'n p h d -> n h p d')
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)
        out = (attn @ v).transpose(1, 2).reshape(-1, C)
        feat = feat + self.proj(out)[unpad]

        x.feat = feat + self.mlp(self.norm2(feat))
        x.sparse_conv_feat = x.sparse_conv_feat.replace_feature(x.feat)
        return x

# ---------- 稀疏全局编码器 ----------
class SparseTransformerGlobal(nn.Module):
    def __init__(self, dim=256, depth=4, n_heads=8, patch_size=128, device=torch.device("cuda")):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.device = device
        self.stem = PointSequential(spconv.SubMConv3d(4, dim, 1, bias=False).to(device),
                                    nn.BatchNorm1d(dim, dim))
        self.blocks = nn.ModuleList([SparseTransformerBlock(dim, n_heads=n_heads, patch_size=patch_size).to(device) for _ in range(depth)])
        self.norm = nn.RMSNorm(dim).to(device)
        self.cls = nn.Parameter(torch.randn(1, dim, device=device))

    def forward(self, point):
        # voxels: [M, 5, 3]  coords: [M, 4] (batch, z, y, x)
        x = self.stem(point)
        M = x.feat.shape[0]  # 当前 batch 体素数
        freqs_cis = precompute_freqs_cis(
            self.dim // self.n_heads,  # head_dim
            M,  # ← 实际长度
        )
        for blk in self.blocks:
            x = blk(x, freqs_cis.to(self.device))
        # 1. 用 reduce_mean 直接按 batch 维度求平均
        x.global_feat = torch_scatter.segment_csr(
            src=x.feat,  # [M, C]
            indptr=torch.cat([x.offset.new_zeros(1), x.offset]),  # [B] 右边界
            reduce='mean',
            out=None)  # 输出 [B, C]
        # 2. 加 cls token 并 LayerNorm
        x.global_feat = self.norm(x.global_feat + self.cls)  # [B, C]
        return x

# ---------- Pose Head ----------
class PoseHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            PointSequential(
                nn.Linear(dim * 2, dim, bias=False),
                nn.GELU(),
                nn.Linear(dim, dim // 2, bias=False),
                nn.GELU(),
                nn.Linear(dim // 2, dim // 2, bias=False),
                nn.GELU(),
                nn.Linear(dim // 2, dim // 4, bias=False),
                nn.GELU(),
                nn.Linear(dim // 4, 7, bias=False),
            )
        )

    def forward(self, g_src, g_tgt):
        fuse = torch.cat([g_src.global_feat, g_tgt.global_feat], dim=1)  # [B, 512]
        pose = self.mlp(fuse)
        t = pose[:, :3]
        q = nn.functional.normalize(pose[:, 3:], dim=1)
        return t, q

# ---------- 完整网络 ----------
@MODELS.register_module("MyModel")
class MyModel(PointModule):
    def __init__(self, voxel_size=0.05, **kwargs):
        super().__init__(**kwargs)
        self.enc = SparseTransformerGlobal()
        self.head = PoseHead()

    def forward(self, data_dict):
        sorce_dict = {
            "coord": data_dict["coord"],
            "grid_coord": data_dict["grid_coord"],
            "offset": data_dict["offset"],
            "feat": data_dict["orig"],
        }
        target_dict = {
            "coord": data_dict["coord_transformed"],
            "grid_coord": data_dict["grid_coord_transformed"],
            "offset": data_dict["offset_trans"],
            "feat": data_dict["trans"],
        }
        sorce_point = Point(sorce_dict)
        target_point = Point(target_dict)

        sorce_point.sparsify()
        target_point.sparsify()

        # 全局特征
        g_src = self.enc(sorce_point)
        g_tgt = self.enc(target_point)

        # 位姿
        t, q = self.head(g_src, g_tgt)
        return t, q