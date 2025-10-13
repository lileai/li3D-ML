# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2023/10/8 14:47
# @Author  : Leii
# @File    : get_camera_view_json.py
# @Code instructions: 
# ------------------------------------------------------------------------
import open3d as o3d
import os
import numpy as np
from os.path import join


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"{filename}", width=1600, height=990)
    vis.add_geometry(pcd)
    vis.run()
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"{filename}", width=1600, height=990)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def get_file_extension(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower()  # 将后缀名转为小写，方便比较


if __name__ == "__main__":
    # 读取点云文件
    # filename = '1683274158.217676640.pcd'
    # filename = '000457.pcd'
    # filename = 'ori_1739522377.259831.pcd'
    # filename = 'Ship_1745939323_0.pcd'
    # filename = 'ship_1749796837.8978858_-39.051231384277344.pcd'
    filename = '0.pcd'
    # filename = '000001.bin'
    # dataset = 'semantickitti'
    # dataset = 'customer_data'
    dataset = 'icp_data'
    save_path = './view'
    extension = get_file_extension(filename)
    if extension == ".pcd":
        pcd = o3d.io.read_point_cloud(f"../data/{dataset}/{filename}")
        pcd = o3d.io.read_point_cloud(f"D:\program\li3D-ML\data\deck_pcd\ship_1755513708.3049033_-40.31502151489258_8017_-40.23896598815918.pcd")
    elif extension == ".bin":
        data_path = f'../data/{dataset}/dataset/sequences/08'
        pc_path = join(data_path, 'velodyne', filename)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.fromfile(pc_path,
                                 dtype=np.float32).reshape((-1, 4))[:, 0:3])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    T_lidar2cam = np.array([
        [0.950728, 0.049246, 0.306089, 0],
        [-0.300211, -0.100213, 0.948594, 0],
        [0.077388, -0.993747, -0.080492, 0],
        [76.342424, 17.336712, -23.733077, 1]
    ])

    # pcd = pcd.transform(T_lidar2cam)
    save_view_point(pcd, f"{save_path}/{dataset}_viewpoint.json")  # 保存好得json文件位置
    load_view_point(pcd, f"{save_path}/{dataset}_viewpoint.json")  # 加载修改时较后的pcd文件
