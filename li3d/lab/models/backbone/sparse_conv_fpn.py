import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch import SubMConv3d, SparseConv3d, SparseInverseConv3d
# from ..builder import MODELS
from spconv.pytorch.utils import PointToVoxel

class SparseConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, indice_key=None,
                 norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm1d
        act_layer = act_layer or nn.ReLU
        if stride == 1:
            self.conv = SubMConv3d(in_ch, out_ch, 3, stride=1, padding=1,
                                   indice_key=indice_key, bias=False)
        else:
            self.conv = SparseConv3d(in_ch, out_ch, 3, stride=stride, padding=1,
                                     indice_key=indice_key, bias=False)
        self.norm = norm_layer(out_ch)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = x.replace_feature(self.act(self.norm(x.features)))
        return x


class SparseResidualBlock(nn.Module):
    def __init__(self, channels, indice_key, norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm1d
        act_layer = act_layer or nn.ReLU
        self.conv1 = SubMConv3d(channels, channels, 3, stride=1, padding=1,
                                indice_key=indice_key, bias=False)
        self.norm1 = norm_layer(channels)
        self.act1 = act_layer()
        self.conv2 = SubMConv3d(channels, channels, 3, stride=1, padding=1,
                                indice_key=indice_key, bias=False)
        self.norm2 = norm_layer(channels)
        self.act2 = act_layer()

    def forward(self, x):
        identity = x
        out = x.replace_feature(self.act1(self.norm1(self.conv1(x).features)))
        out = x.replace_feature(self.norm2(self.conv2(out).features))
        out = out + identity
        out = out.replace_feature(self.act2(out.features))
        return out

# @MODELS.register_module()
class SparseConvFPN(nn.Module):
    def __init__(self, in_channel, out_channel, base_ch=32,
                 norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm1d
        act_layer  = act_layer  or nn.GELU

        # ---------------- encoder -----------------
        # stage-1  (stride=1，纯 subm，可以复用)
        self.s1_init = SparseConvBlock(in_channel, base_ch, stride=1, indice_key="s1_subm")
        self.s1_res  = SparseResidualBlock(base_ch, indice_key="s1_subm")

        # stage-2  下采样 + 两个子流形
        self.s2_down = SparseConvBlock(base_ch, base_ch*2, stride=2, indice_key="s2_down")
        self.s2_res1 = SparseResidualBlock(base_ch*2, indice_key="s2_subm")
        self.s2_res2 = SparseResidualBlock(base_ch*2, indice_key="s2_subm")

        # stage-3
        self.s3_down = SparseConvBlock(base_ch*2, base_ch*4, stride=2, indice_key="s3_down")
        self.s3_res1 = SparseResidualBlock(base_ch*4, indice_key="s3_subm")
        self.s3_res2 = SparseResidualBlock(base_ch*4, indice_key="s3_subm")

        # stage-4
        self.s4_down = SparseConvBlock(base_ch*4, base_ch*8, stride=2, indice_key="s4_down")
        self.s4_res1 = SparseResidualBlock(base_ch*8, indice_key="s4_subm")
        self.s4_res2 = SparseResidualBlock(base_ch*8, indice_key="s4_subm")

        # ---------------- decoder -----------------
        self.up3 = SparseInverseConv3d(base_ch*8, base_ch*4, 3, indice_key="s4_down")  # 与下采样同一 key
        self.dec3 = SparseResidualBlock(base_ch*4, indice_key="d3_subm")

        self.up2 = SparseInverseConv3d(base_ch*4, base_ch*2, 3, indice_key="s3_down")
        self.dec2 = SparseResidualBlock(base_ch*2, indice_key="d2_subm")

        self.up1 = SparseInverseConv3d(base_ch*2, base_ch, 3, indice_key="s2_down")
        self.dec1 = SparseConvBlock(base_ch, out_channel, stride=1, indice_key="d1_subm")

    def forward(self, x):
        x1 = self.s1_res(self.s1_init(x))
        x2 = self.s2_res2(self.s2_res1(self.s2_down(x1)))
        x3 = self.s3_res2(self.s3_res1(self.s3_down(x2)))
        x4 = self.s4_res2(self.s4_res1(self.s4_down(x3)))

        y3 = self.dec3(self.up3(x4) + x3)
        y2 = self.dec2(self.up2(y3) + x2)
        y1 = self.dec1(self.up1(y2) + x1)
        return [y1, y2, y3, x4]


# ---------------- main ----------------
if __name__ == "__main__":
    B, N = 2, 8000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 随机点云
    points_list = []
    for b in range(B):
        xyz = np.random.rand(N, 3) * 50
        intensity = np.random.rand(N, 1)
        batch_idx = np.full((N, 1), b, dtype=np.float32)
        points_list.append(np.hstack([batch_idx, xyz, intensity]))
    points = torch.tensor(np.vstack(points_list), dtype=torch.float32, device=device)

    # 体素化
    gen = PointToVoxel(vsize_xyz=[0.2, 0.2, 0.2],
                       coors_range_xyz=[0, 0, -3, 50, 50, 1],
                       num_point_features=4,
                       max_num_voxels=30000,
                       max_num_points_per_voxel=10,
                       device=device)
    voxels, coords, _ = gen(points)
    voxel_feat = voxels.mean(dim=1)
    batch_idx = torch.full((coords.shape[0], 1), b, dtype=torch.int32, device=coords.device)
    coords = torch.cat([batch_idx, coords], dim=1)  # [M, 4]
    coords = coords.int()
    sparse_tensor = spconv.SparseConvTensor(
        features=voxel_feat,
        indices=coords,
        spatial_shape=gen.grid_size[::-1],  # 新版兼容
        batch_size=B
    )

    # 网络
    net = SparseConvFPN(in_channel=4, out_channel=256, base_ch=32).to(device)
    net.eval()
    with torch.no_grad():
        feats = net(sparse_tensor)

    for lvl, f in enumerate(feats):
        print(f'level{lvl}: {f.features.shape}  voxels={f.features.shape[0]}')