# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2024/11/29 9:38
# @Author  : Leii
# @File    : 引导滤波.py
# @Code instructions: 
# ------------------------------------------------------------------------
import open3d as o3d
import numpy as np
import os
import open3d.visualization.gui as gui
import copy
import glob


def guided_filter(pcd, radius, epsilon):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points_copy = np.array(pcd.points)
    points = np.asarray(pcd.points)
    num_points = len(pcd.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 3:
            continue

        neighbors = points[idx, :]
        mean = np.mean(neighbors, 0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))

        A = cov @ e
        b = mean - A @ mean

        points_copy[i] = A @ points[i] + b

    return points_copy


def add_noise(pcd, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    noise = sigma * np.random.randn(points.shape[0], points.shape[1])
    points += noise
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


if __name__ == '__main__':
    # 读取点云文件
    # filename = '1683274158.217676640.pcd'
    # filename = 'ori_1739522377.259831.pcd'
    filename = '000457.pcd'
    dataset = 'send/ship1'
    # dataset = 'customer_data'
    directory_path = f"../data/{dataset}"
    pcd_files = glob.glob(f'{directory_path}/*.pcd', recursive=True)
    for filename in pcd_files:
        print(filename)
        pcd = o3d.io.read_point_cloud(f"{filename}", format="pcd")
        print("Statistical oulier removal")
        points_copy = guided_filter(pcd, 0.1, 0.01)
        pcd_finl = o3d.geometry.PointCloud()
        pcd_finl.points = o3d.utility.Vector3dVector(points_copy)
        points_copy = guided_filter(pcd_finl, 0.01, 0.1)
        pcd_finl = o3d.geometry.PointCloud()
        pcd_finl.points = o3d.utility.Vector3dVector(points_copy)

        pcd.paint_uniform_color([1, 1, 1])
        pcd_finl.paint_uniform_color([1, 1, 1])

        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("POINT")
        vis.show_settings = True

        vis.add_geometry("pcd_finl", pcd_finl)
        vis.add_geometry("pcd", pcd)

        set_viewer = True if os.path.exists('./view/customer_data_viewpoint.json') else False
        if set_viewer:  # 自定义可视化初始视角参数
            param = o3d.io.read_pinhole_camera_parameters('./view/ship1_viewpoint.json')
            extrinsic_matrix, intrinsic_matrix = param.extrinsic, param.intrinsic
            intrinsic_matrix, height, width = intrinsic_matrix.intrinsic_matrix, intrinsic_matrix.height, intrinsic_matrix.width
            vis.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)
        app.add_window(vis)
        app.run()
