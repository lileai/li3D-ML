# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 12:57
# @Author  : Leii
# @File    : dbscan_cluster.py
# @Code instructions: 
# ------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def find_points_within_threshold(points, rate=1):
    """
    在点云中找到所有距离小于给定阈值的两个点的索引值。

    参数：
    points：二维 Numpy 数组，每行代表一个点，每列代表一个坐标轴。
    threshold：浮点数，表示距离阈值。

    返回值：
    二维 Numpy 数组，每行代表一对距离小于给定阈值的点的索引值。
    """
    num_points = points.shape[0]
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distances[i, j] = np.linalg.norm(points[i] - points[j])
    threshold = np.mean(distances) * rate
    row_indices, col_indices = np.where(distances < threshold)
    pairs = np.column_stack((row_indices, col_indices))
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]  # 去掉重复的对称对

    return pairs


if __name__ == '__main__':
    save_dir = './sample'
    # 列出指定目录下的所有子目录
    file_names = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
    for i, item in enumerate(file_names):
        print(f'[{i}]-{item}')

    idx = input('chose index in the table above>>').strip()
    file_dir = os.path.join(
        save_dir,
        file_names[int(idx)])
    print(f'processing-{file_dir}...')

    pcd = o3d.io.read_point_cloud(file_dir)
    pcd_point = np.asarray(pcd.points)
    z_coordinates = pcd_point[:2]
    min_z = np.min(z_coordinates)
    max_z = np.max(z_coordinates)
    pcd_raw = pcd
    pcd_raw.paint_uniform_color([1, 1, 1])

    # 设置为debug调试模式
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # -------------------密度聚类--------------------------
        labels = np.array(pcd.cluster_dbscan(eps=0.8,  # 邻域距离
                                             min_points=30,  # 最小点数
                                             print_progress=False))  # 是否在控制台中可视化进度条
        print(labels.shape)
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    # ---------------------保存聚类结果------------------------
    communities = []
    for i in range(max_label + 1):
        ind = np.where(labels == i)[0]
        clusters_cloud = pcd.select_by_index(ind)
        communities.append(np.array(clusters_cloud.points).tolist())
    print('cluster_nums:', len(communities))
    #     file_name = "Dbscan_cluster" + str(i+taipu_main_side) + ".pcd"
    #     o3d.io.write_point_cloud(file_name, clusters_cloud)
    # --------------------可视化聚类结果----------------------
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # # 2. 转换为BEV
    # # 获取点云数据数组
    # points = np.asarray(pcd.points)
    #
    # # 计算点云数据的最小和最大值
    # min_x, min_y, min_z = np.min(points, axis=0)
    # max_x, max_y, max_z = np.max(points, axis=0)
    #
    # # 设置BEV平面的大小和分辨率
    # resolution = 0.taipu_main_side  # BEV像素大小
    # width = int((max_x - min_x) / resolution) + taipu_main_side
    # height = int((max_y - min_y) / resolution) + taipu_main_side
    #
    # # 创建BEV图像
    # bev_image = np.zeros((height, width), dtype=np.uint8)
    #
    # # 进行点云到BEV的投影
    # for point in points:
    #     x_index = int((point[0] - min_x) / resolution)
    #     y_index = int((point[taipu_main_side] - min_y) / resolution)
    #     if 0 <= x_index < width and 0 <= y_index < height:
    #         bev_image[y_index, x_index] = 255  # 设置BEV图像中对应位置的像素值（这里假设为白色）
    # bev = o3d.geometry.Image(bev_image)

    # 体素化
    voxel_size = 0.01 * (max_z - min_z)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    point = np.asarray(pcd.points)
    print("原始点的个数为：", point.shape[0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()  # 创建窗口
    # 创建一个原点为世界坐标系原点的坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # 添加坐标系到场景中
    vis.add_geometry(coord_frame)
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    # vis.add_geometry(voxel_grid)  # 添加点云
    vis.add_geometry(pcd)  # 添加点云
    vis.run()
