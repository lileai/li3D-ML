# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 20:43
# @Author  : Leii
# @File    : 统计滤波.py
# @Code instructions: 
# ------------------------------------------------------------------------
import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from matplotlib import cm
from os.path import join


def get_file_extension(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower()  # 将后缀名转为小写，方便比较


if __name__ == '__main__':
    is_intensity = True
    # 读取点云文件
    filename = '1683274158.217676640.pcd'
    # filename = '000457.pcd'
    # filename = 'ori_1739522315.3921025.pcd'
    # filename = '000100.bin'
    # dataset = 'semantickitti'
    dataset = 'customer_data'
    # dataset = '发送/ship1'
    extension = get_file_extension(filename)
    if extension == ".pcd":
        pcd = o3d.io.read_point_cloud(f"../data/{dataset}/{filename}")
        pcd_t = o3d.t.io.read_point_cloud(f"../data/{dataset}/{filename}")
        if is_intensity:
            pcd_intensity = pcd_t.point['intensity']
            intensity = pcd_intensity[:, :].numpy()  # 转换为数组类型
    elif extension == ".bin":
        data_path = f'../data/{dataset}/dataset/sequences/01'
        pc_path = join(data_path, 'velodyne', filename)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.fromfile(pc_path,
                                                            dtype=np.float32).reshape((-1, 4))[:, 0:3])
        if is_intensity:
            intensity = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))[:, 3:]
    # 归一化强度值
    if is_intensity:
        intensities_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        colors = cm.gist_rainbow(intensities_normalized)  # 'viridis'是一个颜色映射名，你可以选择其他映射
        colors = colors[:, :3].squeeze(1)  # 我们只需要RGB，不需要alpha通道
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # voxel_pcd = pcd.voxel_down_sample(voxel_size=0.5)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_pcd, voxel_size=0.5)

    # 转换为Open3D的颜色格式
    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,
                                             std_ratio=2.0)
    sor_cloud = pcd.select_by_index(ind)

    points1 = np.asarray(pcd.points)
    points2 = np.asarray(sor_cloud.points)
    # 创建点云索引
    index1 = set(map(tuple, points1))
    index2 = set(map(tuple, points2))

    # 计算补集点索引
    complement_index = index1 - index2

    # 创建一个新的点云对象，并将交集点添加到其中
    pcd_filter = o3d.geometry.PointCloud()
    pcd_filter.points = o3d.utility.Vector3dVector(np.asarray(list(complement_index)))

    # pcd.paint_uniform_color([taipu_main_side, taipu_main_side, taipu_main_side])
    sor_cloud.paint_uniform_color([1, 1, 1])
    pcd_filter.paint_uniform_color([1, 0, 0])

    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("POINT")
    vis.show_settings = True

    vis.add_geometry("sor_cloud", sor_cloud)
    vis.add_geometry("pcd_filter", pcd_filter)
    # vis.add_geometry("voxel_grid", voxel_grid)
    vis.add_geometry("pcd", pcd)

    set_viewer = True if os.path.exists('./view/customer_data_viewpoint.json') else False
    if set_viewer:  # 自定义可视化初始视角参数
        param = o3d.io.read_pinhole_camera_parameters('./view/customer_data_viewpoint.json')
        extrinsic_matrix, intrinsic_matrix = param.extrinsic, param.intrinsic
        intrinsic_matrix, height, width = intrinsic_matrix.intrinsic_matrix, intrinsic_matrix.height, intrinsic_matrix.width
        vis.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)
    app.add_window(vis)
    app.run()
