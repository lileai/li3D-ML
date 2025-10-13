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
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from ..utils.logger import get_root_logger
from ..utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class DefaultDataset(Dataset):
    # 允许加载的 .npy（特征）名字白名单
    VALID_ASSETS = [
        "coord", "color", "normal", "strength",
        "segment", "instance", "pose",
    ]

    # ---------------- 初始化 ----------------
    def __init__(
        self,
        split="train",              # 划分名：train / val / test
        data_root="data/dataset",   # 数据集根目录
        transform=None,             # 训练/验证阶段数据增强列表
        test_mode=False,            # True→推理模式，走测试增强
        test_cfg=None,              # 测试阶段专属配置（体素化/裁剪/TTA）
        cache=False,                # 是否把样本缓存到内存（加速二次读取）
        ignore_index=-1,            # 分割任务里要忽略的类别 id
        loop=1,                     # 训练时重复几次样本；测试强制 1
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)  # 把增强列表串成 pipeline
        self.cache = cache
        self.ignore_index = ignore_index
        # 测试模式下强制 loop=1，防止 TTA 被重复
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        # -------- 测试模式专属增强 --------
        if test_mode:
            # 测试体素化（稀疏网格化）
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            # 滑动窗口或中心裁剪
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop)
                if self.test_cfg.crop else None
            )
            # 裁剪后归一化、ToTensor 等
            self.post_transform = Compose(self.test_cfg.post_transform)
            # TTA：对同一点云做多次旋转/翻转，后续投票
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # -------- 生成样本清单 --------
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_list), self.loop,
                os.path.basename(self.data_root), split
            )
        )

    # ---------------- 样本清单生成 ----------------
    def get_data_list(self):
        # 1. 允许外部传单个字符串或列表，统一转成列表
        if isinstance(self.split, str):
            split_list = [self.split]  # 例如 "train" → ["train"]
        elif isinstance(self.split, Sequence):  # 例如 ["train", "val"]
            split_list = self.split
        else:
            raise NotImplementedError  # 其他类型直接抛错

        data_list = []  # 最终要返回的「样本路径」列表

        # 2. 逐个 split 处理
        for split in split_list:
            # 2-a 如果 split 本身是 **文件**（例如 train.txt）
            if os.path.isfile(os.path.join(self.data_root, split)):
                with open(os.path.join(self.data_root, split)) as f:
                    # 按 JSON 读取，里面应是 ["scene0000_00", "scene0001_00", ...]
                    data_list += [
                        os.path.join(self.data_root, data)  # 拼成绝对路径
                        for data in json.load(f)
                    ]
            # 2-b 如果 split 是 **目录**（最常见）
            else:
                # 用通配符把 split 目录下所有子文件夹/文件一次性列出来
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))

        # 3. 返回大列表，里面每个元素都是「一个样本」的路径
        return data_list

    # ---------------- 真正读盘 ----------------
    def get_data(self, idx):
        # 支持 loop>1 时的循环索引
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        # 若开启缓存，直接读共享内存（加速）
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            # 只加载白名单里的特征
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))

        # 附加 meta
        data_dict["name"] = name
        data_dict["split"] = split

        # -------- 类型强制转换 --------
        if "coord" in data_dict:
            data_dict["coord"] = data_dict["coord"].astype(np.float32)
        if "color" in data_dict:
            data_dict["color"] = data_dict["color"].astype(np.float32)
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        # 若不存在标签，填 -1
        if "segment" in data_dict:
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if "instance" in data_dict:
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        return data_dict

    # ---------------- 辅助函数 ----------------
    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def get_split_name(self, idx):
        return os.path.basename(
            os.path.dirname(self.data_list[idx % len(self.data_list)])
        )

    # ---------------- 训练样本构造 ----------------
    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)   # 应用训练增强
        return data_dict

    # ---------------- 测试样本构造（TTA + 滑动窗口） ----------------
    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)   # 先整体 transform

        # 把标签 & 名字拿出来，不送进网络
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        # TTA：对同一点云做多次增强
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        # 对每个增强版本做体素化 → 裁剪 → 片段列表
        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)  # 可能返回 list
                else:
                    data_part = [data_part]
                fragment_list += data_part

        # 后处理：归一化、ToTensor 等
        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])

        result_dict["fragment_list"] = fragment_list
        return result_dict

    # ---------------- PyTorch 协议 ----------------
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


# ------------------------------------------------------------------
# 下面是一个 ConcatDataset：把多个 Dataset 链在一起当成一个大数据集
# ------------------------------------------------------------------
@DATASETS.register_module()
class ConcatDataset(Dataset):
    def __init__(self, datasets, loop=1):
        super(ConcatDataset, self).__init__()
        # 递归 build 每个子数据集
        self.datasets = [build_dataset(dataset) for dataset in datasets]
        self.loop = loop
        self.data_list = self.get_data_list()

        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in the concat set.".format(
                len(self.data_list), self.loop
            )
        )

    def get_data_list(self):
        # 生成 [(dataset_idx, sample_idx), ...] 的扁平列表
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i]), dtype=int) * i,
                    np.arange(len(self.datasets[i])),
                )
            )
        return data_list

    def get_data(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx][data_idx]  # 调子数据集的 __getitem__

    def get_data_name(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx].get_data_name(data_idx)

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop