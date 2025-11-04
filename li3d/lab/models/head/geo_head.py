import torch
import math
import torch.nn as nn
from torch.nn.functional import embedding

# from ..utils.registration.local_global_registration import LocalGlobalRegistration
# from ..builder import MODELS
# from ..utils.misc import batch2offset, offset2mask, pairwise_distance, index_select
# geo_head.py
from li3d.lab.models.utils.structure import Point
from li3d.lab.models.utils.registration.local_global_registration import LocalGlobalRegistration
from li3d.lab.models.utils.misc import offset2batch, packed2batch, pairwise_distance, index_select
import spconv.pytorch as spconv


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings


class RelPosEmbedding(nn.Module):
    def __init__(self, dim, sigma_d, sigma_a, angle_k, reduction_a):
        super().__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * torch.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(dim)
        self.proj_d = nn.Linear(dim, dim)
        self.proj_a = nn.Linear(dim, dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        # 确保 points 是整数类型
        assert points.dtype in [torch.int32, torch.int64], "Points should be integer type"

        batch_size, num_point, _ = points.shape

        # Compute pairwise distances
        points_float = points.float()  # 转换为浮点数
        dist_map = torch.cdist(points_float, points_float)  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        # Compute k-nearest neighbors
        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)

        # Compute reference and anchor vectors
        expanded_points = points_float.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points_float.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points_float.unsqueeze(1) - points_float.unsqueeze(2)  # (B, N, N, 3)

        # Compute angles
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        cross_product = torch.cross(ref_vectors, anc_vectors, dim=-1)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(cross_product, dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        # Compute distance embeddings
        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        # Compute angle embeddings
        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        # Combine embeddings
        embeddings = d_embeddings + a_embeddings

        return embeddings


class RPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, embedding=True):
        super(RPEMultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5
        self.embedding = embedding

        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        if self.embedding:
            self.proj_p = nn.Linear(d_model, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        """
        Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, N, C)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        B, N, C = input_q.shape
        _, M, _ = input_k.shape

        # 1. 线性投影 + 拆头
        q = self.proj_q(input_q).view(B, N, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        k = self.proj_k(input_k).view(B, M, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, M, head_dim)
        v = self.proj_v(input_v).view(B, M, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, M, head_dim)
        if self.embedding:
            p = self.proj_p(embed_qk).view(B, N, M, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, H, N, M, head_dim)
            attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q, p)  # (B, H, N, M)
            attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)  # (B, H, N, M)
            attention_scores = (attention_scores_e + attention_scores_p) * self.scale  # (B, H, N, M)
        else:
            attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) * self.scale  # (B, H, N, M)

        # 3. 处理key_masks
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(2),
                                                            float('-inf'))  # (B, H, N, M)
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores  # (B, H, N, M)

        # 4. softmax + dropout
        attention_scores = self.softmax(attention_scores)  # (B, H, N, M)
        attention_scores = self.dropout(attention_scores)  # (B, H, N, M)

        # 5. 加权求和
        hidden_states = torch.matmul(attention_scores, v)  # (B, H, N, head_dim)
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        hidden_states = self.proj_out(hidden_states)  # (B, N, C)

        return hidden_states


class MLP(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels=None,
            out_channels=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_0 = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x) + x_0
        return x


class TransformerLayer(nn.Module):
    def __init__(self,
                 channels,
                 n_head,
                 mlp_ratio=4,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 act_layer=nn.GELU,
                 embedding=True):
        super().__init__()
        self.attn = RPEMultiHeadAttention(channels, n_head, attn_drop, embedding)
        self.norm1 = nn.RMSNorm(channels)
        self.norm2 = nn.RMSNorm(channels)
        self.ffn = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm3 = nn.RMSNorm(channels)

    def forward(self, src, tgt, embedding=None, tgt_mask=None):
        """
        src: (B, N, C)   packed
        tgt: (B, M, C)   packed
        embedding: torch.Tensor (B, N, M, C), relative positional embedding
        *_mask    : (B, L)      True=valid
        return    : (L, B, C)
        """
        # 1. self-attention on target
        src1 = self.attn(src, tgt, tgt, embedding, tgt_mask)
        src = src + self.norm1(src1)

        # 3. FFN
        src2 = self.ffn(src)
        src = src + self.norm3(src2)
        return src


class RPEConditionalTransformer(nn.Module):
    def __init__(self,
                 blocks,
                 channels,
                 n_head,
                 sigma_d,
                 sigma_a,
                 angle_k,
                 reduction_a,
                 mlp_ratio=4,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 act_layer=nn.GELU,
                 return_scores=False, parallel=False):
        super().__init__()
        self.blocks = blocks
        self.parallel = parallel
        self.return_scores = return_scores
        self.rpe = RelPosEmbedding(channels, sigma_d, sigma_a, angle_k, reduction_a)
        layers = []
        for blk in blocks:
            if blk == "self":
                layers.append(TransformerLayer(channels,
                                               n_head,
                                               mlp_ratio,
                                               attn_drop,
                                               proj_drop,
                                               act_layer))
            else:
                layers.append(TransformerLayer(channels,
                                               n_head,
                                               mlp_ratio,
                                               attn_drop,
                                               proj_drop,
                                               act_layer,
                                               embedding=False))
        self.layers = nn.ModuleList(layers)
        self.in_proj = nn.Linear(channels, channels)

    def forward(self, feats0, feats1, coord0, coord1, masks0=None, masks1=None):
        """
        feats*     : (L, B, C)   packed 特征
        coord*: (B, L, 3)   packed 坐标（当 RPE 用）
        masks*     : (B, L)      True=valid
        return     : (L, B, C)   同形状
        """

        feats0 = self.in_proj(feats0)
        feats1 = self.in_proj(feats1)

        embedding_0 = self.rpe(coord0)
        embedding_1 = self.rpe(coord1)

        for i, blk in enumerate(self.blocks):
            if blk == 'self':
                feats0 = self.layers[i](src=feats0, tgt=feats0, embedding=embedding_0, tgt_mask=masks0)
                feats1 = self.layers[i](src=feats1, tgt=feats1, embedding=embedding_1, tgt_mask=masks1)
            else:  # cross
                feats0 = self.layers[i](src=feats0, tgt=feats1, tgt_mask=masks1)
                feats1 = self.layers[i](src=feats1, tgt=feats0, tgt_mask=masks0)

        return feats0, feats1


class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores


@torch.no_grad()
def build_voxel_knn_batch(
        x_fine,
        x_coarse,
        k=32,
        voxel_size=1.0):
    device = x_fine.features.device
    C_feat = x_fine.features.shape[1]

    fine_idx = x_fine.indices  # (N, 4)
    coarse_idx = x_coarse.indices  # (M, 4)

    fine_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                             fine_idx[:, 0].bincount().cumsum(0)], 0)
    coarse_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                               coarse_idx[:, 0].bincount().cumsum(0)], 0)
    B = coarse_offset.size(0) - 1

    m_max = (coarse_offset[1:] - coarse_offset[:-1]).max().item()
    n_max = (fine_offset[1:] - fine_offset[:-1]).max().item()
    k = min(k, n_max)

    knn_indices = torch.full((B, m_max, k), x_fine.features.shape[0],
                             dtype=torch.long, device=device)
    knn_masks = torch.zeros(B, m_max, k, dtype=torch.bool, device=device)
    knn_points = torch.zeros(B, m_max, k, 3, dtype=torch.float, device=device)
    knn_feats = torch.zeros(B, m_max, k, C_feat, dtype=torch.float, device=device)
    # ✅ 对标 node_masks 的低分辨率 mask
    coarse_masks = torch.zeros(B, m_max, dtype=torch.bool, device=device)

    half_win = int(math.ceil(k ** (1 / 3) / 2)) + 1
    delta = torch.arange(-half_win, half_win + 1, device=device)
    delta = torch.stack(torch.meshgrid(delta, delta, delta, indexing='ij'), -1).view(-1, 3)

    for b in range(B):
        c0, c1 = coarse_offset[b], coarse_offset[b + 1]
        f0, f1 = fine_offset[b], fine_offset[b + 1]
        m_frame = c1 - c0
        n_frame = f1 - f0

        coarse_coord = coarse_idx[c0:c1, 1:].long()
        fine_coord = fine_idx[f0:f1, 1:].long()
        fine_feat = x_fine.features[f0:f1]

        local_hash = {}
        for idx, (x, y, z) in enumerate(fine_coord.cpu().numpy()):
            local_hash[(int(x), int(y), int(z))] = idx

        # ✅ 标记本帧真实 voxel 位
        coarse_masks[b, :m_frame] = True

        for m in range(m_frame):
            ctr = coarse_coord[m].cpu().numpy()
            neigh = []
            for d in delta:
                key = (int(ctr[0] + d[0]), int(ctr[1] + d[1]), int(ctr[2] + d[2]))
                if key in local_hash:
                    neigh.append(local_hash[key])
                    if len(neigh) == k:
                        break
            if neigh:
                l = len(neigh)
                idx_tensor = torch.tensor(neigh, device=device)
                knn_indices[b, m, :l] = f0 + idx_tensor
                knn_masks[b, m, :l] = True
                knn_points[b, m, :l] = fine_coord[idx_tensor].float() * voxel_size
                knn_feats[b, m, :l] = fine_feat[idx_tensor]

    return knn_indices, knn_masks, knn_points, knn_feats, coarse_masks


class LearnableLogOptimalTransport(nn.Module):
    def __init__(self, num_iterations, inf=1e12):
        r"""Sinkhorn Optimal transport with dustbin parameter (SuperGlue style)."""
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iterations = num_iterations
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.0)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores, row_masks=None, col_masks=None):
        r"""Sinkhorn Optimal Transport (SuperGlue style) forward.

        Args:
            scores: torch.Tensor (B, M, N)
            row_masks: torch.Tensor (B, M)
            col_masks: torch.Tensor (B, N)

        Returns:
            matching_scores: torch.Tensor (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape

        if row_masks is None:
            row_masks = torch.ones(size=(batch_size, num_row), dtype=torch.bool).cuda()
        if col_masks is None:
            col_masks = torch.ones(size=(batch_size, num_col), dtype=torch.bool).cuda()

        padded_row_masks = torch.zeros(size=(batch_size, num_row + 1), dtype=torch.bool).cuda()
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(size=(batch_size, num_col + 1), dtype=torch.bool).cuda()
        padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
        padded_scores.masked_fill_(padded_score_masks, -self.inf)

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(size=(batch_size, num_row + 1)).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -self.inf

        log_nu = torch.empty(size=(batch_size, num_col + 1)).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -self.inf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iterations={})'.format(self.num_iterations)
        return format_string


# @MODELS.register_module("GEO-HEAD")
class GeoHead(nn.Module):
    def __init__(self, blocks,
                 channels,
                 n_head,
                 sigma_d=0.2,
                 sigma_a=15,
                 angle_k=3,
                 reduction_a='max',
                 mlp_ratio=4,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 act_layer=None,
                 return_scores=False,
                 parallel=False,
                 num_correspondences=False,
                 dual_normalization=False,
                 patch_size=26,  # 一个体素块的最小邻居个数
                 voxel_size=0.2,
                 num_iterations=100,
                 topk=None,
                 acceptance_radius=None,
                 mutual=True,
                 confidence_threshold=0.05,
                 use_dustbin=False,
                 use_global_score=False,
                 correspondence_threshold=3,
                 correspondence_limit=None,
                 num_refinement_steps=5
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.voxel_size = voxel_size
        self.act_layer = act_layer or nn.GELU
        self.transformer = RPEConditionalTransformer(
            blocks=blocks,
            channels=channels,
            n_head=n_head,
            sigma_d=sigma_d,
            sigma_a=sigma_a,
            angle_k=angle_k,
            reduction_a=reduction_a,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            act_layer=self.act_layer,
            return_scores=return_scores,
            parallel=parallel
        )
        self.coarse_matching = SuperPointMatching(
            num_correspondences, dual_normalization
        )
        self.optimal_transport = LearnableLogOptimalTransport(num_iterations=num_iterations)
        self.fine_matching = LocalGlobalRegistration(
            topk,
            acceptance_radius,
            mutual=mutual,
            confidence_threshold=confidence_threshold,
            use_dustbin=use_dustbin,
            use_global_score=use_global_score,
            correspondence_threshold=correspondence_threshold,
            correspondence_limit=correspondence_limit,
            num_refinement_steps=num_refinement_steps,
        )

    def forward(self,
                source_dict,  # 源点云字典
                target_dict,  # 目标点云字典
                source_list,  # 源点云FPN后的特征列表
                target_list):  # 目标点云FPN后的特征列表

        source_dict.feat = source_list[0].features
        source_dict.batch = source_list[0].indices[:, 0]
        source_dict.sp_feat = source_list[-1].features
        source_dict.sp_batch = source_list[-1].indices[:, 0]
        source_dict.sp_coord = source_list[-1].indices[:, 1:]

        target_dict.feat = target_list[0].features
        target_dict.batch = target_list[0].indices[:, 0]
        target_dict.sp_feat = target_list[-1].features
        target_dict.sp_batch = target_list[-1].indices[:, 0]
        target_dict.sp_coord = target_list[-1].indices[:, 1:]
        source_dict.sp_feat, source_dict.sp_coord, masks0 = packed2batch(source_dict.sp_feat, source_dict.sp_coord,
                                                                         source_dict.sp_batch)
        target_dict.sp_feat, target_dict.sp_coord, masks1 = packed2batch(target_dict.sp_feat, target_dict.sp_coord,
                                                                         target_dict.sp_batch)
        source_dict.sp_feat, target_dict.sp_feat = self.transformer(feats0=source_dict.sp_feat,
                                                                    feats1=target_dict.sp_feat,
                                                                    coord0=source_dict.sp_coord,
                                                                    coord1=target_dict.sp_coord,
                                                                    masks0=masks0,
                                                                    masks1=masks1)
        with torch.no_grad():
            # 在低分辨率生成一一对应的匹配点
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                target_dict.sp_feat, source_dict.sp_feat, masks0,
                masks1
            )
            source_dict.corr_indices = src_node_corr_indices
            target_dict.corr_indices = ref_node_corr_indices
        # 求出低分辨率的voxel的K个邻居在高分辨率的voxel中的索引（这里的pts是voxel，在y1中代表点）
        # knn_idx：低分辨率的voxel的K个邻居在高分辨率的voxel中的索引 shape (B, M_max, K)
        # knn_mask：用于区分上述索引中哪些邻居是有效的，哪些是无效的（真实不存在，但是为了批次计算而填充的） shape (B, M_max, K)
        # knn_coord：每个低分辨率voxel的K个邻居的世界坐标 shape (B, M_max, K, 3)
        # knn_feat：每个低分辨率voxel的K个邻居的特征 shape (B, M_max, K, C)
        src_knn_idx, src_knn_mask, src_knn_coord, src_knn_feat, source_dict.sp_mask = build_voxel_knn_batch(
            x_fine=source_list[0],
            x_coarse=source_list[-1],
            k=self.patch_size,
            voxel_size=self.voxel_size)
        ref_knn_idx, ref_knn_mask, ref_knn_coord, ref_knn_feat, target_dict.sp_mask = build_voxel_knn_batch(
            x_fine=target_list[0],
            x_coarse=target_list[-1],
            k=self.patch_size,
            voxel_size=self.voxel_size)
        # 获得低分辨率 匹配voxel 的邻居在高分辨率voxel中的索引（前者是所有低分辨率的voxel，这里只要匹配点的）
        target_dict.corr_knn_indices = ref_knn_idx[ref_node_corr_indices]  # (P, K)
        source_dict.corr_knn_indices = src_knn_idx[src_node_corr_indices]  # (P, K)

        target_dict.corr_knn_mask = ref_knn_mask[ref_node_corr_indices]  # (P, K)
        source_dict.corr_knn_mask = src_knn_mask[src_node_corr_indices]  # (P, K)

        target_dict.corr_knn_coord = ref_knn_coord[ref_node_corr_indices]  # (P, K, 3)
        source_dict.corr_knn_coord = src_knn_coord[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feat = torch.cat([target_dict.feat, torch.zeros_like(target_dict.feat[:1])], dim=0)
        src_padded_feat = torch.cat([source_dict.feat, torch.zeros_like(source_dict.feat[:1])], dim=0)
        target_dict.corr_knn_feat = index_select(ref_padded_feat, target_dict.corr_knn_indices, dim=0)  # (P, K, C)
        source_dict.corr_knn_feat = index_select(src_padded_feat, source_dict.corr_knn_indices, dim=0)  # (P, K, C)

        matching_scores = torch.einsum('bnd,bmd->bnm', target_dict.corr_knn_feats,
                                       source_dict.corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / source_dict.feat.shape[1] ** 0.5
        # 求解最优传输问题
        matching_scores = self.optimal_transport(matching_scores, target_dict.corr_knn_mask,
                                                 source_dict.corr_knn_mask)  # (P, K+1, K+1)
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            target_dict.corr_knn_point, source_dict.corr_knn_point, corr_scores, estimated_transform = self.fine_matching(
                target_dict.corr_knn_coord,
                source_dict.corr_knn_coord,
                target_dict.corr_knn_mask,
                source_dict.corr_knn_mask,
                matching_scores,
                node_corr_scores,
            )

        return source_dict, target_dict, corr_scores, estimated_transform


def fake_fpn_no_backbone(batch_size=2, base_ch=32, device="cuda"):
    """
    返回 4 级稀疏张量，shape 与 SparseConvFPN 完全一致
    stride: [1, 2, 4, 8]
    channel: [base_ch, base_ch*2, base_ch*4, base_ch*8]
    """
    stride = [1, 2, 4, 8]
    ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
    fpn = []
    for s, c in zip(stride, ch):
        # 1. 随机 voxel 数
        n = torch.randint(400, 800, (1,)).item()

        # 2. 随机网格坐标
        indices = torch.randint(0, 128 // s, (n, 3), device=device)

        # 3. 构造 batch 索引 —— 严格长度 == n
        step = n // batch_size
        batch = torch.arange(batch_size, device=device).repeat_interleave(step)
        # 补余数
        rem = n - batch.numel()
        if rem:
            batch = torch.cat([batch, torch.full((rem,), batch_size - 1, device=device)])

        # 4. 拼接成 (N,4) 并转 int
        indices = torch.cat([batch.view(-1, 1), indices], 1).int()
        assert indices.shape[0] == n, "batch 维度与坐标维度必须一致"

        # 5. 随机特征
        feats = torch.randn(n, c, device=device)

        # 6. 构造稀疏张量
        tensor = spconv.SparseConvTensor(
            feats, indices,
            spatial_shape=[128 // s] * 3,
            batch_size=batch_size
        )
        fpn.append(tensor)
    return fpn


# ---------- 伪造输入字典 ----------
def make_fake_input_no_backbone(batch_size=2, device="cuda"):
    """只造 GeoHead 需要的字段，不再建 Point 对象"""
    n = torch.randint(2000, 3000, (1,)).item()
    coord = torch.randn(n, 3, device=device) * 10
    R = torch.linalg.qr(torch.randn(3, 3, device=device))[0]
    t = torch.randn(3, device=device)
    coord_trans = coord @ R.T + t
    voxel_size = 0.2
    grid_coord = torch.floor(coord / voxel_size).long()
    grid_coord_trans = torch.floor(coord_trans / voxel_size).long()
    batch = torch.arange(batch_size, device=device).repeat_interleave(n // batch_size)
    offset = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                        torch.bincount(batch).cumsum(0)], 0)
    return dict({
        "orig": torch.randn(n, 1, device=device),
        "trans": torch.randn(n, 1, device=device),
        "coord": coord,
        "coord_transformed": coord_trans,
        "grid_coord": grid_coord,
        "grid_coord_transformed": grid_coord_trans,
        "offset": offset,
        "offset_trans": offset,
        "transform_matrix": torch.eye(4, device=device).view(1, 4, 4).repeat(batch_size, 1, 1)
    })


# ---------- 主测试 ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. 伪造 FPN 输出
    src_list = fake_fpn_no_backbone(device=device)
    tgt_list = fake_fpn_no_backbone(device=device)
    print("伪造 FPN 形状（features）:", [t.features.shape for t in src_list])

    # 2. 伪造输入字典
    src_dict = make_fake_input_no_backbone(device=device)
    tgt_dict = make_fake_input_no_backbone(device=device)

    src_dict = Point(src_dict)
    tgt_dict = Point(tgt_dict)

    # 3. 实例化 GeoHead（配置同前）
    cfg = dict(
        blocks=['self', 'cross', 'self', 'cross'],
        channels=256,
        n_head=8,
        num_correspondences=128,
        dual_normalization=True,
        num_iterations=10,
        voxel_size=0.2,
        patch_size=26,
        mlp_ratio=4,
        attn_drop=0.1,
        proj_drop=0.1
    )
    head = GeoHead(**cfg).to(device)

    # 4. 前向
    with torch.no_grad():
        src_dict, tgt_dict, scores, T = head(src_dict, tgt_dict, src_list, tgt_list)

    print("匹配对数:", scores.shape[0])
    print("估计变换:\n", T.cpu())
