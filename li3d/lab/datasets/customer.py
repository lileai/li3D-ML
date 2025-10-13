"""
Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import json
from re import split

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from copy import deepcopy
from collections.abc import Sequence

from ..utils.logger import get_root_logger
from ..utils.cache import shared_dict
from .defaults import DefaultDataset
from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class CustomerDataset(DefaultDataset):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            if os.path.isfile(os.path.join(self.data_root, split)):
                with open(os.path.join(self.data_root, split)) as f:
                    data_list += [
                        os.path.join(self.data_root, data) for data in json.load(f)
                    ]
            else:
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        # ---- 1. 读 PCD ----
        pcd = o3d.io.read_point_cloud(data_path)
        points = np.asarray(pcd.points)  # (N,3)
        if pcd.has_colors():  # 强度存在 color 通道
            intensity = np.asarray(pcd.colors)[:, 2].reshape(-1, 1)  # 取 R 通道当 intensity
        else:
            intensity = np.ones(len(points)).reshape(-1, 1)

        # ---- 2. 随机齐次变换（旋转 + 平移） ----
        R = Rotation.random().as_matrix()  # 3×3 随机旋转
        t = np.random.uniform(-0.5, 0.5, size=3)  # 随机平移
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t  # 4×4 齐次矩阵

        # ---- 3. 应用变换 ----
        ones = np.ones((len(points), 1))
        pts_h = np.hstack([points, ones])  # (N,4)
        pts_transformed = (T @ pts_h.T).T[:, :3]  # (N,3)

        # ---- 4. 打包返回 ----
        data_dict = {
            "coord": points.astype(np.float32),  # 原始 XYZ
            "intensity": intensity.astype(np.float32),  # 原始 I
            "coord_transformed": pts_transformed.astype(np.float32),  # 变换后 XYZ
            "intensity_transformed": intensity.astype(np.float32),  # 同一强度
            "transform_matrix": T.astype(np.float32),  # 4×4 标签
            "index_valid_keys": ["coord", "intensity"],
            "index_valid_keys_trans": ["coord_transformed", "intensity_transformed"],
            "name": name,
            "split": split,
        }

        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def get_split_name(self, idx):
        return os.path.basename(
            os.path.dirname(self.data_list[idx % len(self.data_list)])
        )
