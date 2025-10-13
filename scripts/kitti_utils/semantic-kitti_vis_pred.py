# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 19:35
# @Author  : Leii
# @File    : semantic-kitti_vis_pred.py
# @Code instructions:
# ------------------------------------------------------------------------
import logging
import os
import torch
import time
from os.path import join, dirname, abspath
from serialization import encode

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import yaml
import glob

example_dir = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


class GridSample(object):
    def __init__(
            self,
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            keys=("coord", "color", "normal", "segment"),
            return_inverse=False,
            return_grid_coord=False,
            return_min_coord=False,
            return_displacement=False,
            project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "point" in data_dict.keys()
        scaled_coord = data_dict["point"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                        scaled_coord - grid_coord - 0.5
                )  # [0, taipu_main_side] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                            scaled_coord - grid_coord - 0.5
                    )  # [0, taipu_main_side] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


def get_file_extension(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower()  # 将后缀名转为小写，方便比较


def load_pc_kitti(pc_path, pc_name):
    extension = get_file_extension(pc_name)
    if extension == ".pcd":
        pcd = o3d.t.io.read_point_cloud(pc_path)
        pcd_points = pcd.point["positions"]  # 坐标
        pcd_intensity = pcd.point["intensity"]  # 强度
        # pcd_ring = pcd.point["ring"]  # 环数
        # pcd_time = pcd.point["time"]  # 时间戳
        pcd_points = pcd_points[:, :].numpy()  # 转换为数组类型
        pcd_intensity = pcd_intensity[:, :].numpy()  # 转换为数组类型
        # pcd_ring = pcd_ring[:, :].numpy()  # 转换为数组类型
        # pcd_time = pcd_time[:, :].numpy()  # 转换为数组类型
        scan = np.concatenate((pcd_points, pcd_intensity), axis=-1)
        return scan
    elif extension == ".bin":
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan


def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def get_custom_data(pc_name,
                    path,
                    data_config_file,
                    decimals=None,
                    label_path=None,
                    infer=False,
                    map=False,
                    label_file="labels"):
    data_config = join(dirname(abspath(__file__)), data_config_file)
    DATA = yaml.safe_load(open(data_config, 'r'))
    remap_dict = DATA["learning_map_inv"]

    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

    remap_dict_val = DATA["learning_map"]
    max_key = max(remap_dict_val.keys())
    remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

    pc_path = join(path, 'velodyne', pc_name)
    points = load_pc_kitti(pc_path, pc_name)

    if decimals is not None:  # 需要量化数据
        points = np.around(points, decimals=decimals)

    if not infer:  # 推理的时候标签是网络计算出来的，不是文件给的
        if label_path is None:
            label_path = join(path, label_file, pc_name.rsplit(".", 1)[0] + ".label")
        else:
            label_path = label_path
        labels = load_label_kitti(label_path, remap_lut_val).astype(np.int32)
        data = {
            'point': points[:, 0:3],
            'feat': points[:, 3:],
            'label': labels,
        }
        grid_sample = GridSample(grid_size=0.05,
                                 hash_type="fnv",
                                 mode="train",
                                 keys=("point", "feat", "label"),
                                 return_grid_coord=True, )
    else:
        data = {
            'point': points[:, 0:3],
            'feat': points[:, 3:],
        }
        # grid_sample = GridSample(grid_size=0.05,
        #                          hash_type="fnv",
        #                          mode="train",
        #                          keys=("point", "feat"),
        #                          return_grid_coord=True, )

    if map:
        # return grid_sample(data), DATA['color_map'], DATA['learning_map']
        return data, DATA['color_map'], DATA['learning_map']
    else:
        # return grid_sample(data)
        return data


def main():
    color_map = {
        0: [0, 0, 0],  # "unlabeled", and others ignored
        1: [100, 150, 245],  # "car"
        2: [100, 230, 245],  # "bicycle"
        3: [30, 60, 150],  # "motorcycle"
        4: [80, 30, 180],  # "truck"
        5: [0, 0, 255],  # "other-vehicle"
        6: [255, 30, 30],  # "person"
        7: [255, 40, 200],  # "bicyclist"
        8: [150, 30, 90],  # "motorcyclist"
        9: [255, 0, 255],  # "road"
        10: [255, 150, 255],  # "parking"
        11: [75, 0, 75],  # "sidewalk"
        12: [175, 0, 75],  # "other-ground"
        13: [255, 200, 0],  # "building"
        14: [255, 120, 50],  # "fence"
        15: [0, 175, 0],  # "vegetation"
        16: [135, 60, 0],  # "trunk"
        17: [150, 240, 80],  # "terrain"
        18: [255, 240, 150],  # "pole"
        19: [255, 0, 0],  # "traffic-sign"
    }
    label_map = {
        0: 'unlabeled, and others ignored',  # "unlabeled", and others ignored
        1: "car",  # "car"
        2: "bicycle",  # "bicycle"
        3: "motorcycle",  # "motorcycle"
        4: "truck",  # "truck"
        5: "other-vehicle",  # "other-vehicle"
        6: "person",  # "person"
        7: "bicyclist",  # "bicyclist"
        8: "motorcyclist",  # "motorcyclist"
        9: "road",  # "road"
        10: "parking",  # "parking"
        11: "sidewalk",  # "sidewalk"
        12: "other-ground",  # "other-ground"
        13: "building",  # "building"
        14: "fence",  # "fence"
        15: "vegetation",  # "vegetation"
        16: "trunk",  # "trunk"
        17: "terrain",  # "terrain"
        18: "pole",  # "pole"
        19: "traffic-sign",  # "traffic-sign"
    }
    # data_path = '../data/semantickitti/dataset/sequences/08'
    data_path = '../../data/ours/sequences/01'
    pc_names = '000000.bin'
    # 批量读取指定路径下的pcd文件名
    pcd_files = glob.glob(f'{data_path}/velodyne/*.bin', recursive=True)
    # 提取文件名
    pcd_file_names = [os.path.basename(p) for p in pcd_files]
    # 初始化app界面(静态)
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("POINT", width=1920, height=1080)
    vis.show_settings = True
    vis.show_skybox(False)
    set_viewer = True if os.path.exists('../view/semantickitti_viewpoint.json') else False
    if set_viewer:  # 自定义可视化初始视角参数
        param = o3d.io.read_pinhole_camera_parameters('../view/semantickitti_viewpoint.json')
        extrinsic_matrix, intrinsic_matrix = param.extrinsic, param.intrinsic
        intrinsic_matrix, height, width = intrinsic_matrix.intrinsic_matrix, intrinsic_matrix.height, intrinsic_matrix.width
        vis.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)
    vis.point_size = 1
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None
    app.add_window(vis)
    for filename in pcd_file_names:
        # pcs, color_map, label_map = get_custom_data(filename, data_path, 'semantic-kitti-our.yaml', map=True)
        pcs, color_map, label_map = get_custom_data(filename, data_path, 'semantic-kitti-our.yaml', map=True, label_file="predictions")
        # pcs, color_map, label_map = get_custom_data(pc_names, data_path, 'semantic-kitti-our.yaml', map=True)
        # pcs = get_custom_data(pc_names, data_path, 'semantic-kitti.yaml', map=False)
        # pcs_2 = get_custom_data(pc_names, data_path, 'semantic-kitti.yaml', map=False, label_path="./test/000000_2.label")
        label = pcs['label']
        # label_2 = pcs_2['label']
        # code = encode(torch.tensor(pcs['grid_coord']), None, depth=int(pcs['grid_coord'].max()).bit_length(), order='hilbert')
        # order = torch.argsort(code).detach().numpy()
        # label_order = pcs['label'][order]
        # idices = np.arange(label_order.shape[0])
        # idx_shift_right = np.roll(idices, taipu_main_side, axis=0)
        # idx_shift_left = np.roll(idices, -taipu_main_side, axis=0)
        # idx_shift_right_2 = np.roll(idices, 2, axis=0)
        # idx_shift_left_2 = np.roll(idices, -2, axis=0)
        # labels_right = label_order[idx_shift_right]
        # labels_left = label_order[idx_shift_left]
        # labels_right_2 = label_order[idx_shift_right_2]
        # labels_left_2 = label_order[idx_shift_left_2]
        # difference = ((label_order != labels_right) | (label_order != labels_left) | (label_order != labels_right_2) | (label_order != labels_left_2)).astype(int)
        # difference = (label != label_2).astype(int)
        # print(np.unique(difference, return_index=True))
        # color_diff = np.zeros([len(difference), 3])
        # for i, cla in enumerate(difference):
        #     if cla == 0:
        #         color_diff[i][0] = 255
        #         color_diff[i][taipu_main_side] = 255
        #         color_diff[i][2] = 255
        #     else:
        #         color_diff[i][0] = 255
        #         color_diff[i][taipu_main_side] = 0
        #         color_diff[i][2] = 0

        colors = np.zeros([len(label), 3])
        for cla in range(len(label)):
            value = color_map[label[cla]]
            colors[cla][0] = value[2]
            colors[cla][1] = value[1]
            colors[cla][2] = value[0]
            # colors[cla][0] = value[0]
            # colors[cla][taipu_main_side] = value[taipu_main_side]
            # colors[cla][2] = value[2]
        # df = pd.DataFrame(colors)
        # print(df.value_counts(sort=True))
        class_dict = {}
        for i in np.unique(label):
            class_pcd = o3d.geometry.PointCloud()
            classes = np.where(label == i)
            class_pcd.points = o3d.utility.Vector3dVector(pcs['point'][classes])
            class_colors = colors[classes] / 255.0
            class_pcd.colors = o3d.utility.Vector3dVector(class_colors)
            class_dict[label_map[i]] = class_pcd

        # pcd_diff = o3d.geometry.PointCloud()
        # # pcd_diff.points = o3d.utility.Vector3dVector(pcs_2['point'])
        # pcd_diff.points = o3d.utility.Vector3dVector(pcs['point'][order][difference != 0])
        # pcd_diff.colors = o3d.utility.Vector3dVector((color_diff[difference == taipu_main_side] / 255.0))
        #
        # pcd_raw = o3d.geometry.PointCloud()
        # # pcd_diff.points = o3d.utility.Vector3dVector(pcs_2['point'])
        # pcd_raw.points = o3d.utility.Vector3dVector(pcs['point'][order])
        # pcd_raw.paint_uniform_color([taipu_main_side, taipu_main_side, taipu_main_side])

        all_pcd = o3d.geometry.PointCloud()
        all_pcd.points = o3d.utility.Vector3dVector(pcs['point'])
        colors_all = colors / 255.0
        all_pcd.colors = o3d.utility.Vector3dVector(colors_all)
        print('点云个数：', np.asarray(all_pcd.points).shape[0])

        # 更新几何体
        vis.remove_geometry("all")  # 移除旧的点云
        vis.add_geometry("all", all_pcd)  # 添加新的点云

        # 强制重绘以更新视图
        vis.post_redraw()

        # 这里可以添加一个短暂的延时以便观察变化，例如：

        # for class_label, pcd in class_dict.items():
        #     vis.add_geometry(f'{class_label}', pcd)
        # vis.add_geometry("pcd", pcd_raw)
        # vis.add_geometry("diff", pcd_diff)
        # if class_label == 6:
        #     o3d.io.write_point_cloud("./土堆.pcd", pcd)
        # for class_label, pcd in class_dict.items():
        #     vis.add_3d_label(pcd.points[0], f'{class_label}')
        time.sleep(0.1)
    app.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()
