# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 19:53
# @Author  : Leii
# @File    : knn.py
# @Code instructions: 
# ------------------------------------------------------------------------
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
import open3d.visualization.gui as gui


def spherical_to_cartesian(position):
    x = position[:, 0] * np.sin(position[:, 1]) * np.cos(position[:, 2])
    y = position[:, 0] * np.sin(position[:, 1]) * np.sin(position[:, 2])
    z = position[:, 0] * np.cos(position[:, 1])
    return np.column_stack((x, y, z))


if __name__ == '__main__':
    # 读取点云文件
    # filename = '1683274158.217676640.pcd'
    filename = '000009.pcd'
    # filename = '000009_curvature_pcd_0.12.pcd'
    dataset = 'customer_data'
    pcd = o3d.io.read_point_cloud(f"../data/{dataset}/{filename}")
    # pcd = o3d.io.read_point_cloud(f"./sample/{filename}")
    point = np.asarray(pcd.points)
    # 将直角坐标系转换为球坐标系
    r = np.sqrt(point[:, 0] ** 2 + point[:, 1] ** 2 + point[:, 2] ** 2)  # 计算 r
    theta = np.arccos(point[:, 2] / r)  # 计算 theta
    phi = np.arctan2(point[:, 1], point[:, 0])  # 计算 phi
    spherical_point = np.column_stack((r, theta, phi))

    # # 手动进行归一化
    # r_min, r_max = np.min(r), np.max(r)
    # theta_min, theta_max = np.min(theta), np.max(theta)
    # phi_min, phi_max = np.min(phi), np.max(phi)
    #
    # r_normalized = (r - r_min) / (r_max - r_min)
    # theta_normalized = (theta - theta_min) / (theta_max - theta_min)
    # phi_normalized = (phi - phi_min) / (phi_max - phi_min)
    #
    # spherical_point_normalized = np.column_stack((r_normalized, theta_normalized, phi_normalized))

    print(point[0].reshape(1, -1))

    kdtree = KDTree(point)
    knn_idx = kdtree.query(np.array([0, 0, 0]).reshape(1, -1), k=100)[1]

    spherical_kdtree = KDTree(spherical_point)
    knn_idx_sph = spherical_kdtree.query(np.array([0, 0, 0]).reshape(1, -1), k=100)[1]

    knn_point = point[knn_idx].squeeze(0)
    knn_pcd = o3d.geometry.PointCloud()
    knn_pcd.points = o3d.utility.Vector3dVector(knn_point)

    knn_point_sph = spherical_point[knn_idx_sph].squeeze(0)

    # # 反归一化
    # r_denormalized = knn_point_sph[:, 0] * (r_max - r_min) + r_min
    # theta_denormalized = knn_point_sph[:, taipu_main_side] * (theta_max - theta_min) + theta_min
    # phi_denormalized = knn_point_sph[:, 2] * (phi_max - phi_min) + phi_min
    #
    # spherical_point_denormalized = np.column_stack((r_denormalized, theta_denormalized, phi_denormalized))

    knn_point_sph2cart = spherical_to_cartesian(knn_point_sph)
    knn_pcd_sph = o3d.geometry.PointCloud()
    knn_pcd_sph.points = o3d.utility.Vector3dVector(knn_point_sph2cart)

    knn_pcd.paint_uniform_color([0, 0, 1])
    knn_pcd_sph.paint_uniform_color([1, 0, 0])

    pcd_center = o3d.geometry.PointCloud()
    pcd_center.points = o3d.utility.Vector3dVector(np.array([0, 0, 0]).reshape(1, -1))
    pcd_center.paint_uniform_color([0, 1, 0])

    # 初始化app界面
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("POINT")
    vis.show_settings = True
    vis.add_geometry("pcd", pcd)
    vis.add_geometry("knn_points", knn_pcd)
    vis.add_geometry("knn_points_sph", knn_pcd_sph)
    vis.add_geometry("center", pcd_center)

    app.add_window(vis)
    app.run()
