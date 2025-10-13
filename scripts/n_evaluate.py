# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2023/6/10 19:53
# @Author  : Leii
# @File    : n_evaluate.py
# @Code instructions: 基于法向量估计的点云采样
# ------------------------------------------------------------------------
import open3d as o3d
import numpy as np
import os
import open3d.visualization.gui as gui


def vison(*args):
    pcds = []
    for item in args:
        if type(item) is np.ndarray:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(item)[:, 0:3])
            pcds.append(pcd)
        else:
            pcds.append(item)
    vis = o3d.visualization.Visualizer()
    vis.create_window()  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    color = [[0, 0, 1], [1, 0, 0]]
    for i, v in enumerate(pcds):
        v.paint_uniform_color(color[i])
        vis.add_geometry(v)  # 添加点云
    vis.run()


def pca_compute(data, sort=True):
    """
     SVD分解计算点云的特征值与特征向量
    :param data: 输入数据
    :param sort: 是否将特征值特征向量进行排序
    :return: 特征值与特征向量
    """
    average_data = np.mean(data, axis=0)  # 求均值
    decentration_matrix = data - average_data  # 去中心化
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值、特征向量

    if sort:
        sort = eigenvalues.argsort()[::-1]  # 降序排列
        eigenvalues = eigenvalues[sort]  # 索引

    return eigenvalues


def caculate_surface_curvature(cloud, radius=0.5):
    """
    计算点云的表面曲率
    :param cloud: 输入点云
    :param radius: k近邻搜索的半径，默认值为：0.5m
    :return: 点云中每个点的表面曲率
    """
    points = np.asarray(cloud.points)
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    num_points = len(cloud.points)
    curvature = []  # 储存表面曲率
    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)

        neighbors = points[idx, :]
        w = pca_compute(neighbors)  # w为特征值
        delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0)
        curvature.append(delt)
    curvature = np.array(curvature, dtype=np.float64)
    return curvature


if __name__ == '__main__':
    # 读取点云文件
    filename = '1683274158.217676640.pcd'
    # filename = '000009.pcd'
    dataset = 'customer_data'
    pcd = o3d.io.read_point_cloud(f"../data/{dataset}/{filename}")
    point = np.asarray(pcd.points)
    z_avg = np.mean(point[:, 2])
    if z_avg >= 0:
        z_avg = 0.1 * z_avg
    else:
        z_avg = 9 * z_avg
    point_process = point[np.where(point[:, 2] <= z_avg)]
    # 平均高度往上是不会出现地面的
    pcd_no_process = o3d.geometry.PointCloud()
    pcd_no_process.points = o3d.utility.Vector3dVector(point[np.where(point[:, 2] > z_avg)])
    # 平均高度往下用于地面计算
    pcd_process = o3d.geometry.PointCloud()
    pcd_process.points = o3d.utility.Vector3dVector(point_process)
    surface_curvature = caculate_surface_curvature(pcd_process, radius=0.5)
    # 法向量计算是针对所有点的
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100))
    normal = np.asarray(pcd.normals)
    nz = normal[:, 2]
    nz_mean = np.mean(nz)
    # 角度阈值参数
    rate = 0.12
    angle_raid = np.mean(surface_curvature) * rate
    # 计算曲率高的点和曲率低的点
    point_high = point_process[np.where(surface_curvature > angle_raid)]
    point_low = point_process[np.where(surface_curvature <= angle_raid)]
    # 区分立面点和地面点
    point_fac = point[np.where(abs(nz) < 0.2)]
    point_pla = point[np.where(abs(nz) > 0.99)]

    pcd_high = o3d.geometry.PointCloud()
    pcd_high.points = o3d.utility.Vector3dVector(point_high)

    pcd_low = o3d.geometry.PointCloud()
    pcd_low.points = o3d.utility.Vector3dVector(point_low)

    pcd_fac = o3d.geometry.PointCloud()
    pcd_fac.points = o3d.utility.Vector3dVector(point_fac)

    pcd_pla = o3d.geometry.PointCloud()
    pcd_pla.points = o3d.utility.Vector3dVector(point_pla)

    pcd_no_ground_finl = pcd_no_process + pcd_high + pcd_fac  # 有用的信息点被认为是高度超过阈值的点 + 高曲率的点 + 立面点
    unique_points = np.unique(pcd_no_ground_finl.points, axis=0)
    pcd_no_ground_finl.points = o3d.utility.Vector3dVector(unique_points)
    # 使用numpy的intersect1d函数找到两个点云之间的交集点
    # 将点云对象转换为numpy数组
    points1 = np.asarray(pcd.points)
    points2 = np.asarray(pcd_no_ground_finl.points)
    # 创建点云索引
    index1 = set(map(tuple, points1))
    index2 = set(map(tuple, points2))

    # 计算补集点索引
    complement_index = index1 - index2

    # 创建一个新的点云对象，并将交集点添加到其中
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(np.asarray(list(complement_index)))
    path = './sample'
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = f'{path}/{os.path.splitext(filename)[0]}_curvature_pcd_{rate}.pcd'
    # o3d.io.write_point_cloud(file_name, pcd_no_ground_finl)
    o3d.io.write_point_cloud(file_name, pcd_no_ground_finl)
    pcd_finl_test = o3d.io.read_point_cloud(file_name)
    finl_point_size = np.asarray(pcd_no_ground_finl.points).shape[0]
    print("原始点的个数为：", point.shape[0])
    print("下采样后点的个数为：", finl_point_size)
    pcd.paint_uniform_color([0, 0, 1])
    pcd_ground.paint_uniform_color([0, 0, 0])
    pcd_no_ground_finl.paint_uniform_color([1, 0, 0])
    pcd_high.paint_uniform_color([1, 0, 1])
    pcd_fac.paint_uniform_color([1, 0, 0])

    # 初始化app界面
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("POINT")
    vis.show_settings = True
    vis.add_geometry("pcd", pcd)
    vis.add_geometry("non ground", pcd_no_ground_finl)
    vis.add_geometry("ground", pcd_ground)
    vis.add_geometry("high", pcd_high)
    vis.add_geometry("fac", pcd_fac)
    vis.add_geometry("low", pcd_low)
    vis.add_geometry("pla", pcd_pla)
    app.add_window(vis)
    app.run()
    # vison(pcd_finl_test)
