import open3d as o3d
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from memory_profiler import profile

def voxel_filter(pcd, voxel_size=0.1):
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return down_pcd

def select_center(pcd, return_index=False):
    """
    从点云中筛选出满足特定条件的点：
    taipu_main_side. Z 值小于 Z 轴的下四分位点。
    2. X 值在 X 轴的均值 ±20 范围内的点。
    :param pcd: Open3D 点云对象
    :return: 筛选后的点云对象
    """
    points = np.asarray(pcd.points)
    # 计算 Z 轴的下四分位点
    z_q1 = np.percentile(points[:, 2], 25)  # 25% 分位点
    # 筛选 Z 轴小于下四分位点的点
    z_filter_indices = np.where(points[:, 2] < z_q1)[0]
    x_mean = points[z_filter_indices, 0].mean()
    # 筛选 X 轴在均值 ±20 范围内的点
    x_filter_indices = np.where((points[:, 0] >= (x_mean - 15)) & (points[:, 0] <= (x_mean + 15)))[0]
    process_cloud = pcd.select_by_index(x_filter_indices)
    if return_index:
        return process_cloud, x_filter_indices
    else:
        return process_cloud

def pca_compute(data, sort=True):
    """
     SVD分解计算点云的特征值
    :param data: 输入数据
    :param sort: 是否将特征值进行排序
    :return: 特征值
    """
    average_data = np.mean(data, axis=0)  # 求均值
    decentration_matrix = data - average_data  # 去中心化
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H
    _, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值

    if sort:
        sort = eigenvalues.argsort()[::-1]  # 降序排列
        eigenvalues = eigenvalues[sort]  # 索引
    return eigenvalues

def freeboard_segmentation(pcd, radius=1.5, threshold=0.5):
    """
    计算每一个点的线性特征，并根据线性特征提取线点云
    :param pcd: 输入点云
    :param threshold: 线特征阈值
    :return: 线点云和线之外的点云
    """

    points = np.asarray(pcd.points)
    # 计算每个点的特征值
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    num_points = len(pcd.points)
    linear = []  # 储存线性特征
    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)

        neighbors = points[idx, :]
        w = pca_compute(neighbors)  # w为特征值

        l1 = w[0]  # 点云的特征值lamda1
        l2 = w[1]  # 点云的特征值lamda2
        L = np.divide((l1 - l2), l1, out=np.zeros_like((l1 - l2)), where=l1 != 0)
        linear.append(L)

    linear = np.array(linear)
    # 设置阈值提取线性部分
    idx = np.where(linear > threshold)[0]
    line_cloud_ = pcd.select_by_index(idx)  # 提取点线上的点

    return line_cloud_

def select_largest_connected_component(pcd, radius=1.5):
    """
    从点云中选择最大的连通域
    :param pcd: Open3D 点云对象
    :param radius: 邻域搜索的半径
    :return: 最大的连通域的点云对象
    """
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 构建邻接矩阵
    num_points = len(points)
    adj_matrix = np.zeros((num_points, num_points), dtype=int)
    for i in range(num_points):
        _, indices, _ = kdtree.search_radius_vector_3d(points[i], radius)
        for j in indices:
            if i != j:
                adj_matrix[i, j] = 1

    # 使用图的连通性分析
    graph = csr_matrix(adj_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # 计算每个连通域的大小
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_label = unique_labels[np.argmax(counts)]  # 最大的连通域的标签

    # 选择最大的连通域
    largest_indices = np.where(labels == largest_label)[0]
    largest_component_pcd = pcd.select_by_index(largest_indices)
    return largest_component_pcd

def statistical_outlier(pcd, nb_neighbors, std_ratio):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    clean_pcd = pcd.select_by_index(ind)
    return clean_pcd

def compute_principal_direction(pcd):
    """
    计算点云的主方向
    :param pcd: open3d.geometry.PointCloud 对象
    :return: 主方向向量
    """
    points = np.asarray(pcd.points)
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_direction = eigenvectors[:, np.argmax(eigenvalues)]

    # 确保主方向向量的方向与全局 x 轴方向夹角不超过90度
    reference_direction = np.array([1, 0, 0])
    if np.dot(principal_direction, reference_direction) < 0:
        principal_direction = -principal_direction

    # 创建线段的起点和终点
    start_point = np.mean(np.asarray(pcd.points), axis=0)
    end_point = start_point + principal_direction * 100

    # 创建线段
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start_point, end_point])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
    return principal_direction, line_set

def build_rotation_matrix(principal_direction):
    """
    构建将主方向对齐到 x 轴的旋转矩阵，仅绕 y 轴和 z 轴旋转
    :param principal_direction: 主方向向量
    :return: 旋转矩阵
    """
    # 归一化主方向向量
    principal_direction = principal_direction / np.linalg.norm(principal_direction)
    # 计算绕 y 轴的旋转角度
    angle_y = np.arctan2(principal_direction[2], principal_direction[0])
    # 构建绕 y 轴的旋转矩阵
    rotation_matrix_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    # 计算绕 z 轴的旋转角度
    # 先将主方向向量绕 y 轴旋转到 xz 平面上
    rotated_principal_direction = np.dot(rotation_matrix_y, principal_direction)
    angle_z = np.arctan2(rotated_principal_direction[1], rotated_principal_direction[0])
    # 构建绕 z 轴的旋转矩阵
    rotation_matrix_z = np.array([
        [np.cos(angle_z), np.sin(angle_z), 0],
        [-np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    # 组合旋转矩阵
    rotation_matrix = np.dot(rotation_matrix_z, rotation_matrix_y)

    return rotation_matrix

def point_cloud_average_density(pcd: o3d.geometry.PointCloud) -> float:
    """
    计算点云的平均密度 (points / m^3)
    :param pcd: Open3D 点云对象
    :return: 平均密度 (float)
    """
    points = np.asarray(pcd.points)
    if points.size == 0:
        return 0.0

    # 包围盒
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    volume = np.prod(max_bound - min_bound)  # (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    # 避免除以零
    if volume <= 1e-12:
        return float('inf')  # 体积过小，视为无限密度

    return len(points) / volume

def apply_rotation(pcd, center_pcd, density):
    """
    应用旋转矩阵到点云（优化版 - 保持原方法不变）
    :param pcd: open3d.geometry.PointCloud 对象
    :param density: 点云密度（points/m³）
    :return: (rotated_pcd, line_set, linear_pcd, linear_pcd_clean, linear_pcd_clean_filter)
    """
    # taipu_main_side. 提取船体线性部分
    linear_pcd = freeboard_segmentation(center_pcd)

    # 2. 密度较低时执行完整滤波流程
    if density <= 0.6:
        # 合并多次滤波步骤
        linear_pcd_clean = select_largest_connected_component(linear_pcd)

        # 连续两次统计滤波 + 连通域提取
        linear_pcd_clean_filter = statistical_outlier(linear_pcd_clean, 20, 1.0)
        linear_pcd_clean_filter = statistical_outlier(linear_pcd_clean_filter, 20, 1.0)

        # 计算主方向
        principal_direction, line_set = compute_principal_direction(linear_pcd_clean_filter)
    else:
        # 高密度时使用center_pcd（假设已在外部定义）
        principal_direction, line_set = compute_principal_direction(center_pcd)

    # 3. 构建旋转矩阵并应用
    rotation_matrix = build_rotation_matrix(principal_direction)
    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)
    rotated_points = np.dot(points - center, rotation_matrix.T) + center

    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)

    # 4. 统一返回5个值
    return rotated_pcd

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

def process_single_ship(points, nums=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = voxel_filter(pcd, voxel_size=0.5)
    # 导向滤波
    points = guided_filter(pcd, radius=1.5, epsilon=0.02)
    pcd.points = o3d.utility.Vector3dVector(points)
    # 取船中点云
    center_pcd = select_center(pcd)
    density = point_cloud_average_density(center_pcd)
    segment_rotation = apply_rotation(pcd, center_pcd, density)
    # 取旋转后的船中
    segment_rotation_center, center_index = select_center(segment_rotation, return_index=True)
    # 导向滤波
    rotation_center_points = guided_filter(segment_rotation_center, radius=1.5, epsilon=0.02)
    guid_segment_rotation_center = o3d.geometry.PointCloud()
    guid_segment_rotation_center.points = o3d.utility.Vector3dVector(rotation_center_points)
    # 提取y坐标
    y_coords = rotation_center_points[:, 1]
    # 找到 y 值最小的20个点
    min_y_top_index =  np.argsort(abs(y_coords))[:int(nums * density)]
    # 对其z轴大小排序
    max_z_idx_in_top = np.argmax( rotation_center_points[min_y_top_index][:, 2])
    global_max_z_idx = min_y_top_index[max_z_idx_in_top]
    rotation_center_points = np.asarray(segment_rotation_center.points)
    freeboard_upper_bound = rotation_center_points[global_max_z_idx, 2] # 获取该点的 z 坐标作为干舷的上界
    freeboard_lower_bound = np.asarray(segment_rotation_center.points)[:, 2].min()
    deck = freeboard_upper_bound - freeboard_lower_bound
    return (
            freeboard_upper_bound,
            freeboard_lower_bound,
            deck
    )


# if __name__ == '__main__':
#     (
#         freeboard_upper_bound,
#         freeboard_lower_bound,
#         deck,
#     ) = process_single_ship(points=points)
