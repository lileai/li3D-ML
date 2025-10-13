"""
双向干舷法干舷计算
"""
import open3d as o3d
import numpy as np
import os
import open3d.visualization.gui as gui
import time
from scipy.spatial import KDTree
from scipy.stats import entropy
import glob
import sys
import matplotlib.pyplot as plt
from collections import Counter

from memory_profiler import profile

def select_center(points, return_index=False):
    """
    取船重心部分
    :param points: 点云
    :return: 筛选后的点云
    """

    # 将点云投射到xoy平面上
    xoy_points = points.copy()
    xoy_points[:, 2] = 0
    min_bound = xoy_points.min(axis=0)
    max_bound = xoy_points.max(axis=0)
    range = abs(max_bound[0] - min_bound[0]) / 6
    point_center  = (min_bound + max_bound) / 2
    # 筛选 X 轴在均值 ±20 范围内的点
    x_filter_indices = np.where((points[:, 0] >= (point_center[0] - range)) & (points[:, 0] <= (point_center[0] + range)))[0]
    process_points = points[x_filter_indices]
    if return_index:
        return process_points, x_filter_indices
    else:
        return process_points

def statistical_outlier(points, nb_neighbors=16, std_ratio=1.0, is_inliers=False):
    """统计滤波去除离群点"""
    # 构建 KDTree
    tree = KDTree(points)

    # 计算每个点的平均距离
    distances, _ = tree.query(points, k=nb_neighbors)
    mean_distances = np.mean(distances, axis=1)

    # 计算标准差
    mean_distance = np.mean(mean_distances)
    stddev = np.std(mean_distances)

    # 找到离群点
    inliers = np.abs(mean_distances - mean_distance) < std_ratio * stddev
    if is_inliers:
        return inliers
    else:
        return points[inliers]


def compute_principal_direction(points):
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    principal = vecs[:, np.argmax(vals)]
    if np.dot(principal, [1, 0, 0]) < 0:
        principal = -principal
        # 对 vecs.T 中的每个特征向量进行方向调整
    vecs_T = vecs.T
    for i in range(vecs_T.shape[0]):
        if np.dot(vecs_T[i], [1, 0, 0]) < 0:
            vecs_T[i] = -vecs_T[i]
    return principal, vals, vecs_T

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


def compute_average_density(points):
    if points.size == 0:
        return 0.0
    min_b, max_b = points.min(axis=0), points.max(axis=0)
    vol = np.prod(max_b - min_b)
    return len(points) / vol if vol > 1e-12 else float('inf')

def estimate_normals(points, radius, max_nn):
    """
    自定义法向量估计函数，使用 Open3D 的 KDTreeSearchParamHybrid。
    :param points: 点云数据，形状为 (N, 3) 的 numpy 数组。
    :param radius: 搜索半径。
    :param max_nn: 最大邻近点数。
    :return: 法向量，形状为 (N, 3) 的 numpy 数组。
    """
    # 将点云数据转换为 Open3D 的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 构建 KDTree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # 初始化法向量数组
    normals = np.zeros_like(points)
    # 遍历每个点，计算其法向量
    for i in range(points.shape[0]):
        # 查询第 i 个点的邻近点
        [_, idx, _] = pcd_tree.search_hybrid_vector_3d(points[i], radius, max_nn)
        # 获取邻域点
        neighbors = np.asarray(pcd.points)[idx]
        # 计算邻域点的均值
        mean = np.mean(neighbors, axis=0)
        # 计算协方差矩阵
        centered_points = neighbors - mean
        covariance = np.dot(centered_points.T, centered_points) / len(idx)
        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # 选择最小特征值对应的特征向量作为法向量
        normals[i] = eigenvectors[:, np.argmin(eigenvalues)]

    return normals

def compute_roughness_entropy(points,
                              k_neighbors=8,
                              radius=1.0):
    # ---------- taipu_main_side. 预处理 ----------
    points = np.asarray(points, dtype=np.float64)
    # 去 NaN/Inf
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    # 去重
    points = np.unique(points, axis=0)
    if points.shape[0] == 0:
        raise ValueError("输入点云为空或全为 NaN/Inf！")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # ---------- 3. 法向量估计 ----------
    normals = estimate_normals(points, radius=radius, max_nn=k_neighbors)
    normals = np.asarray(normals)

    # ---------- 4. 计算粗糙度 ----------
    roughness = np.zeros(len(pcd.points))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for i in range(len(pcd.points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k_neighbors)
        if len(idx) < 3:          # 邻居太少，跳过
            continue
        # 用所有法向量方向的总体标准差作为粗糙度
        roughness[i] = np.linalg.norm(np.std(normals[idx, :], axis=0))

    # ---------- 5. 计算熵 ----------
    total = roughness.sum()
    if total == 0:
        return 0.0
    probs = roughness / total
    probs = probs[probs > 0]        # 避免 log(0)
    roughness_entropy = entropy(probs, base=2)
    return roughness_entropy

def apply_rotation(points, center_points, density):
    """
    应用旋转矩阵到点云（优化版 - 保持原方法不变）
    :param points: 点云
    :return:
    """
    xoy_points = points.copy()
    # 将点云投射到xoy平面上
    xoy_points[:, 2] = 0
    unique_xoy_points = np.unique(xoy_points, axis=0)

    if len(points) < 15000 and density < 0.8:  # 15000有可能也是一半船
        # 将船按几何中心分成两部分
        min_bound = unique_xoy_points.min(axis=0)
        max_bound = unique_xoy_points.max(axis=0)
        xoy_center = (min_bound + max_bound) / 2
        xoy_lower = unique_xoy_points[unique_xoy_points[:, 0] < xoy_center[0]]
        xoy_higher = unique_xoy_points[unique_xoy_points[:, 0] >= xoy_center[0]]
        density_lower = compute_average_density(xoy_lower)
        density_higher = compute_average_density(xoy_higher)
        # 根据密度大小选择部分
        if density_lower < density_higher:
            selected_part = xoy_lower
        else:
            selected_part = xoy_higher

    else:
        selected_part = xoy_points

    principal_direction, vals, vecs_T = compute_principal_direction(selected_part)

    # 3. 构建旋转矩阵并应用
    rotation_matrix = build_rotation_matrix(principal_direction)
    center = np.mean(center_points, axis=0)
    rotated_points = np.dot(center_points - center, rotation_matrix.T) + center
    rotation_total_points = np.dot(points - center, rotation_matrix.T) + center

    return rotation_total_points, rotated_points, principal_direction, xoy_points, rotation_matrix, center, vals, vecs_T, selected_part

def reverse_rotation(rotated_points, rotation_matrix, center, offset=0.0):
    """
    将旋转后的点反变换回原始坐标系
    :param rotated_points: 旋转后的点云
    :param rotation_matrix: 旋转矩阵
    :param center: 旋转中心（保持不变）
    :param offset: 事先人为加上的 y 方向平移量
    :return: 反变换后的点云
    """
    # taipu_main_side. 先减掉“抬高后”的旋转中心
    shifted_points = rotated_points - (center + [0, offset, 0])  # 用临时抬高后的中心
    # 2. 逆旋转
    reversed_points = np.dot(shifted_points, rotation_matrix)
    # 3. 加回“原始”旋转中心（不动）
    original_points = reversed_points + (center + [0, offset, 0])

    return original_points

def divide_points(points, axis=0, n=3):
    # 根据指定轴坐标对点云进行排序，并获取排序后的索引
    sorted_indices = np.argsort(points[:, axis])
    sorted_points = points[sorted_indices]

    # 计算分位点的索引
    total_points = len(points)
    if total_points < n:
        raise ValueError(f"点云中点的数量不足以分成 {n} 等分")

    # 计算每个分组的大小
    group_size = total_points // n
    remainder = total_points % n

    # 分组
    groups = []
    start_index = 0
    for i in range(n):
        # 如果有余数，将余数均匀分配到前面的组中
        end_index = start_index + group_size + (1 if i < remainder else 0)
        group = sorted_points[start_index:end_index]
        groups.append(group)
        start_index = end_index

    return groups


def find_most_frequent_value(arr, precision=1, n=1):
    # 将数组中的每个元素保留到指定的小数位数
    rounded_arr = np.round(arr, precision)

    # 使用 Counter 统计每个元素的出现次数
    value_counts = Counter(rounded_arr)

    # 找到出现次数最多的元素
    most_common_value, _ = value_counts.most_common(1)[0]

    # 找到所有与众数相对应的原始值
    mask = np.round(arr, precision) == most_common_value
    corresponding_values = arr[mask]
    if len(corresponding_values) < n / 5:
        print("true")
        return None

    # 计算这些值的平均值
    mean_value = np.mean(corresponding_values) if len(corresponding_values) > 0 else None

    return mean_value


def filter_by_y(points):
    """
    按 y 值滤波：
    taipu_main_side. 计算 y_min, y_max，取中点作为几何中心 y_center
    2. 计算 y 方向宽度 width = y_max - y_min
    3. 保留 |y - y_center| > width / 3 的点
    返回：过滤后的点云 (M, 3)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) array")

    y = points[:, 1]
    y_min, y_max = y.min(), y.max()
    y_center = (y_min + y_max) / 2.0
    width = y_max - y_min

    mask = np.abs(y - y_center) > (width / 3.0)
    pos_cnt = np.sum(y > 0)
    if pos_cnt > 0.75 * len(y):
        mask_1 = mask & ((y - y_center) < 0).astype(bool)  # 靠近坐标轴的一侧
        mask_2 = mask & ((y - y_center) > 0).astype(bool)  # 远离坐标轴的一侧
        if abs(len(points[mask_1]) - len(points[mask_2])) > 200:
            return points[mask_1], 1, points[mask_2], 2
    else:
        mask_1 = mask & ((y - y_center) > 0).astype(bool)  # 靠近坐标轴的一侧
        mask_2 = mask & ((y - y_center) < 0).astype(bool)  # 远离坐标轴的一侧
    return points[mask_1], -1, points[mask_2], 1

def process_single_ship(points, nums=1, water_level=None):
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    x_center = (x_max + x_min) / 2
    print(x_center)
    if x_center < 50.0 or x_center > 135.0:
        return np.nan, np.nan, np.nan
    center_points = select_center(points)
    density = compute_average_density(center_points)
    if density != float('inf'):
        center_points = select_center(points)
        roughness = compute_roughness_entropy(center_points, radius=1.0 * (1 / density) ** (1/3))
    else:
        return np.nan, np.nan, np.nan
    (
        rotation_total_points,
        rotation_center_points,
        principal_direction,
        xoy_points,
        rotation_matrix,
        center,  # 旋转中心
        vals,
        vecs_T,
        selected_part,
    ) = apply_rotation(points, center_points, density)
    # 将rotation_center_points按x值分成n等分
    n = max(1, int(len(rotation_center_points) / 400))
    print(n)
    y_1, direction_1, y_2, direction_2 = filter_by_y(rotation_center_points)
    print(len(y_1), len(y_2))
    # ---------- 6. 计算 freeboard_upper_bounds ----------
    freeboard_upper_bounds = []
    divide_rotation_center_lists = []
    for y, direction in ((y_1, direction_1), (y_2, direction_2)):
        # 6.taipu_main_side 统计离群点过滤
        yoz_points = y.copy()
        yoz_points[:, 1] = 0
        nb_neighbors = 20 * density
        std_ratio = 0.15 * roughness
        mask = statistical_outlier(yoz_points,
                                   nb_neighbors=nb_neighbors,
                                   std_ratio=std_ratio,
                                   is_inliers=True)
        process_points = y[mask]
        divide_rotation_center_list = divide_points(process_points, axis=0, n=n)
        freeboard_upper_points = []
        for divide_rotation_center in divide_rotation_center_list:
        # 对每个部分提取y坐标
            y_coords = divide_rotation_center[:, 1] + 1000
            y_top_index = np.argsort(direction * np.abs(y_coords))[:max(1, int(nums * roughness))]
            # 对其z轴大小排序
            max_z_idx_in_top = np.argmax(divide_rotation_center[y_top_index][:, 2])
            global_max_z_idx = y_top_index[max_z_idx_in_top]
            freeboard_upper_point = divide_rotation_center[global_max_z_idx]  # 获取该点的 z 坐标作为干舷的上界
            rotation_center_points = np.vstack([rotation_center_points, freeboard_upper_point])
            freeboard_upper_points.append(freeboard_upper_point)
         # 拿出freeboard_upper_points保留两位小数后频数最多的点作为freeboard_upper_point
        divide_rotation_center_list = np.vstack(divide_rotation_center_list + freeboard_upper_points)
        freeboard_upper_points = np.asarray(freeboard_upper_points)
        freeboard_upper_points = reverse_rotation(freeboard_upper_points, rotation_matrix, center)
        freeboard_upper_bound = find_most_frequent_value(freeboard_upper_points[:, 2], n=n)
        if freeboard_upper_bound is None:
            freeboard_upper_bound = np.median(freeboard_upper_points[:, 2])
        freeboard_upper_bounds.append((freeboard_upper_bound))
        divide_rotation_center_lists.append(divide_rotation_center_list)
    if water_level is not None:
        freeboard_lower_bound = water_level
        deck = abs(freeboard_upper_bounds[0] - freeboard_lower_bound)
    else:
        freeboard_lower_bound = center_points[:, 2].min()
        deck = freeboard_upper_bounds[0] - freeboard_lower_bound
    return (
            center_points,
            density,
            roughness,
            principal_direction,
            xoy_points,
            center,
            rotation_total_points,
            rotation_center_points,
            divide_rotation_center_lists[0],
            divide_rotation_center_lists[1],
            freeboard_upper_bounds[0],
            freeboard_upper_bounds[1],
            freeboard_lower_bound,
            deck,
            int(nums * roughness),
            n,
            freeboard_upper_points,
            vals,
            vecs_T,
            selected_part,
    )

def points2pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def split_filename_from_path(file_path):
    """
    从完整路径中提取文件名（不包括扩展名），并按 '_' 分隔为多个部分。
    参数: file_path (str): 完整的文件路径。
    返回: list: 文件名的各个部分。
    """
    # 提取文件名（包括扩展名）
    file_name_with_extension = os.path.basename(file_path)
    # 去掉扩展名
    file_name, _ = os.path.splitext(file_name_with_extension)
    # 使用 split() 方法按 '_' 分隔文件名
    parts = file_name.split('_')
    # 去除空字符串部分（如果存在）
    parts = [part for part in parts if part]

    return parts

def find_first_platform_and_subsequent_averages(z_coordinates):
    # 计算差分
    window_size = max(3, len(z_coordinates) // 10)
    diff = np.diff(z_coordinates)
    threshold = np.mean(np.abs(diff)) * 0.5
    print(f"threshold：{threshold}")
    # 找到第一个显著变化的点
    change_points = np.where(np.abs(diff) > threshold)[0]
    if change_points.size == 0:
        return None

    first_change_index = change_points[0] + 1  # 加1是因为差分导致索引偏移

    # 初始化平台起始索引和平台点列表以及索引列表
    platform_points = []
    platform_indices = []

    # 滑动窗口找到平台
    i = first_change_index
    while i < len(z_coordinates):
        window = z_coordinates[i:i + window_size]
        if len(window) >= window_size and np.std(window) < threshold:
            platform_points.extend(window)
            platform_indices.extend(range(i, min(i + window_size, len(z_coordinates))))
            i += window_size
        else:
            i += 1  # 移动到下一个点

    # 如果没有找到任何平台点，返回 None
    if not platform_points:
        print("找不到平台点")
        return None
    platform = np.mean(platform_points)
    return platform

def build_line(start, direction, value, color=None):
    # 创建线段
    if color is None:
        color = [1, 0, 0]
    end = start + direction * value
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start, end])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])  # 红色
    return line_set


def build_box(points):
    """
    构建并绘制三维点云的包围盒。

    参数:
    points: 三维点云，形状为 (n, 3) 的 NumPy 数组。
    """
    # 计算包围盒的最小和最大坐标
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    # 定义包围盒的8个顶点
    vertices = np.array([
        [min_x, min_y, min_z],  # 0
        [max_x, min_y, min_z],  # taipu_main_side
        [max_x, max_y, min_z],  # 2
        [min_x, max_y, min_z],  # 3
        [min_x, min_y, max_z],  # 4
        [max_x, min_y, max_z],  # 5
        [max_x, max_y, max_z],  # 6
        [min_x, max_y, max_z]  # 7
    ])

    # 定义包围盒的12条边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 竖直边
    ]

    # 创建线对象
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(vertices)
    box.lines = o3d.utility.Vector2iVector(edges)  # 使用 edges 的索引对
    box.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(edges))])  # 设置为红色

    return box


def angle_with_x_axis(vector):
    if len(vector) == 2:  # 二维向量
        x, y = vector
        angle = np.arctan2(y, x)
    elif len(vector) == 3:  # 三维向量
        x, y, z = vector
        angle = np.arctan2(y, x)  # 只考虑 x 和 y 分量
    else:
        raise ValueError("向量必须是二维或三维的")

    return np.degrees(angle)

def make_xoy_plane_through_point(point, size=20.0, color=[0.3, 0.7, 0.9]):
    """
    构造一个过给定点的、平行于 XOY 平面的矩形平面。

    参数
    ----
    point : array_like, shape (3,)
        平面必须经过的点的坐标 [x, y, z]。
    size : float, optional
        正方形平面的边长（中心在 point），默认 5.0。
    color : list or tuple, optional
        平面颜色 RGB，范围 [0,taipu_main_side]，默认 [0.3, 0.7, 0.9]。

    返回
    ----
    open3d.geometry.TriangleMesh
        生成的平面网格。
    """
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("point 必须是长度为 3 的数组")

    # 构建顶点：以 point 为中心
    half = size / 2.0
    vertices = np.array([
        [point[0] - half, point[1] - half, point[2]],
        [point[0] + half, point[1] - half, point[2]],
        [point[0] + half, point[1] + half, point[2]],
        [point[0] - half, point[1] + half, point[2]],
    ], dtype=np.float64)

    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()  # 可选：便于光照
    return mesh

def make_yoz_plane_through_point(point, size=20.0, color=[0.3, 0.7, 0.9]):
    """
    构造一个过给定点的、平行于 YOZ 平面的矩形平面。

    参数
    ----
    point : array_like, shape (3,)
        平面必须经过的点的坐标 [x, y, z]。
    size : float, optional
        正方形平面的边长（中心在 point），默认 20.0。
    color : list or tuple, optional
        平面颜色 RGB，范围 [0,taipu_main_side]，默认 [0.3, 0.7, 0.9]。

    返回
    ----
    open3d.geometry.TriangleMesh
        生成的平面网格。
    """
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("point 必须是长度为 3 的数组")

    half = size / 2.0
    # 顶点以 point 为中心，x 固定，y、z 变化
    vertices = np.array([
        [point[0], point[1] - half, point[2] - half],
        [point[0], point[1] + half, point[2] - half],
        [point[0], point[1] + half, point[2] + half],
        [point[0], point[1] - half, point[2] + half],
    ], dtype=np.float64)

    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


if __name__ == '__main__':
    o3d.utility.random.seed(42)
    # dataset = 'error'
    dataset = 'LL'
    directory_path = f"../data/{dataset}"
    # 获取目标路径下的所有文件和目录
    entries = os.listdir(directory_path)
    # 筛选出第一级目录
    dirs = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]
    d = 0
    for dir in dirs[d:]:
        # 构建当前目录的完整路径
        frame_freeboard_upper_bound = []
        pcd_files = glob.glob(f'{directory_path}/{dir}/*.pcd', recursive=True)
        file_path = None
        n = 0
        for i, pcd_file in enumerate(pcd_files[n:], start=n):
            s = time.time()

            file_path = rf"E:\0723\16_16_01_48_822b261c-a7dc-4745-b8ac-9f90680d80d6_730\detail\deck_pcd\ship_1752652946.2866652_-41.693824768066406_14241.pcd"
            if file_path is not None:
                pcd = o3d.io.read_point_cloud(file_path, format="pcd")
                l = split_filename_from_path(file_path)
            else:
                pcd = o3d.io.read_point_cloud(pcd_file, format="pcd")
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:int(l[-1])])
            (
                center_points,
                density,
                roughness,
                principal_direction,
                xoy_points,
                center,
                rotation_total_points,
                rotation_center_points,
                filter_rotation_center_points_1,
                filter_rotation_center_points_2,
                freeboard_upper_bound_1,
                freeboard_upper_bound_2,
                freeboard_lower_bound,
                deck,
                nums,
                n,
                freeboard_upper_points,
                vals,
                vecs_T,
                selected_part,
            ) = process_single_ship(np.asarray(pcd.points)[:int(l[-1])])
            # ) = process_single_ship(np.asarray(pcd.points), s=s)
            print(f"运行时间：{time.time() - s}")

            # 构建open3d对象
            center_pcd = points2pcd(np.vstack([center_points, freeboard_upper_points]))
            center_pcd.paint_uniform_color([1, 0, 0])
            colors = center_pcd.colors
            for i in range(1, n + 1):
                colors[-i] = [0, 1, 0]
            center_pcd.colors = o3d.utility.Vector3dVector(colors)

            xoy_pcd = points2pcd(xoy_points)
            selected_pcd = points2pcd(selected_part)

            print(angle_with_x_axis(principal_direction))

            # 创建线段的起点和终点
            line_set = []
            start_point = (xoy_points.min(axis=0) + xoy_points.max(axis=0)) / 2
            for direction, value in zip(vecs_T, vals):
                if value == 0.0:
                    continue
                line_set.append(build_line(start_point, direction, value))

            box = build_box(xoy_points)

            # 点云投影代码（没用）
            # 创建新的点云坐标数组，并将所有点的 x 坐标设置为 0
            rotation_center_pcd = points2pcd(rotation_center_points)
            rotation_center_pcd.paint_uniform_color([0, 0, 1])
            colors = rotation_center_pcd.colors
            for i in range(1, n + 1):
                colors[-i] = [0, 1, 0]
            rotation_center_pcd.colors = o3d.utility.Vector3dVector(colors)

            rotation_total_pcd = points2pcd(rotation_total_points)

            filter_rotation_center_pcd_1 = points2pcd(filter_rotation_center_points_1)
            filter_rotation_center_pcd_1.paint_uniform_color([1, 0, 0])
            colors = filter_rotation_center_pcd_1.colors
            for i in range(1, n + 1):
                colors[-i] = [0, 1, 0]
            filter_rotation_center_pcd_1.colors = o3d.utility.Vector3dVector(colors)

            filter_rotation_center_pcd_2 = points2pcd(filter_rotation_center_points_2)
            filter_rotation_center_pcd_2.paint_uniform_color([1, 0, 0])
            colors = filter_rotation_center_pcd_2.colors
            for i in range(1, n + 1):
                colors[-i] = [0, 1, 0]
            filter_rotation_center_pcd_2.colors = o3d.utility.Vector3dVector(colors)

            new_points = np.asarray(rotation_center_pcd.points).copy()
            new_points[:, 0] = 0

            # 创建新的点云对象
            new_pcd = points2pcd(new_points)
            new_pcd.colors = rotation_center_pcd.colors

            platform_x = make_yoz_plane_through_point(center)
            center[2] = freeboard_upper_bound_1
            platform_1 = make_xoy_plane_through_point(center, color=[1, 0, 0])
            center[2] = freeboard_upper_bound_2
            platform_2 = make_xoy_plane_through_point(center, color=[0, 1, 0])

            # ui（没用）
            app = gui.Application.instance
            app.initialize()
            vis = o3d.visualization.O3DVisualizer(f"第{i}帧点云 - {pcd_file if file_path is None else os.path.basename(file_path)} - density：{density} - roughness：{roughness} - 选取{nums}个点 - 上边界高度为：{freeboard_upper_bound_1}, 下边界高度为：{freeboard_lower_bound}, 干舷高度为：{deck}", width=1660, height=900)
            vis.show_skybox(False)
            vis.show_settings = True
            # 设置背景颜色为黑色
            background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
            vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None

            # vis.add_geometry("pcd", pcd)
            # vis.add_geometry("center_pcd", center_pcd)
            # vis.add_geometry("xoy_pcd", xoy_pcd)
            # vis.add_geometry("selected_pcd", selected_pcd)
            # vis.add_geometry("box", box)
            # for i, line in enumerate(line_set):
            #     vis.add_geometry(f"line_{i}", line)
            vis.add_geometry("rotation_total_pcd", rotation_total_pcd)
            # vis.add_geometry("rotation_center_pcd", rotation_center_pcd)
            vis.add_geometry("filter_rotation_center_pcd_1", filter_rotation_center_pcd_1)
            vis.add_geometry("filter_rotation_center_pcd_2", filter_rotation_center_pcd_2)
            # vis.add_geometry("new_pcd", new_pcd)
            vis.add_geometry("platform_1", platform_1)
            vis.add_geometry("platform_2", platform_2)
            # vis.add_geometry("platform_x", platform_x)


            # set_viewer = True if os.path.exists('./view/customer_data_viewpoint.json_files') else False
            set_viewer = False
            if set_viewer:  # 自定义可视化初始视角参数
                param = o3d.io.read_pinhole_camera_parameters('./view/0618_viewpoint.json')
                extrinsic_matrix, intrinsic_matrix = param.extrinsic, param.intrinsic
                intrinsic_matrix, height, width = intrinsic_matrix.intrinsic_matrix, intrinsic_matrix.height, intrinsic_matrix.width
                vis.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)
            app.add_window(vis)
            app.run()
            if file_path is not None:
                sys.exit(0)  # 退出程序，返回状态码 0 表示正常退出
