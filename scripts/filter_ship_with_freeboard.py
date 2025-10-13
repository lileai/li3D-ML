"""
干舷法计算干舷
"""
import os
import glob
import sys
import time
import json
import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import entropy


def voxel_filter(points, voxel_size=0.5):
    """
    点云体素滤波。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :param voxel_size: 体素大小，默认为 0.5。
    :type voxel_size: float
    :return: 体素滤波后的点云。
    :rtype: np.ndarray
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down_pcd.points)


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
    point_center = (min_bound + max_bound) / 2
    # 筛选 X 轴在均值 ±20 范围内的点
    x_filter_indices = \
        np.where((points[:, 0] >= (point_center[0] - range)) & (points[:, 0] <= (point_center[0] + range)))[0]
    process_points = points[x_filter_indices]
    if return_index:
        return process_points, x_filter_indices
    else:
        return process_points


def statistical_outlier(points, nb_neighbors=30, std_ratio=1.0, is_inliers=False):
    """统计滤波去除离群点"""
    # 构建 KDTree
    tree = cKDTree(points)

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


def compute_average_density(points, k_neighbors=8, eps=1e-8):
    points = np.asarray(points)
    if points.size == 0:
        return 0.0
    if points.shape[0] < k_neighbors:
        # 邻居数比点数大，直接返回 NaN 或退化为 k = n
        k_neighbors = points.shape[0]
        if k_neighbors == 0:
            return 0.0
        if k_neighbors == 1:
            # 只有一个点，密度无意义
            return np.nan

    tree = cKDTree(points)
    # query 的 k 从 taipu_main_side 开始编号，因此要 k_neighbors+taipu_main_side 才能把“自己”也算进去
    distances, _ = tree.query(points, k=k_neighbors + 1)
    kth_dist = distances[:, -1] + eps   # 避免 0
    avg_density = k_neighbors / np.mean(kth_dist)
    return avg_density


def compute_roughness_entropy(points, k_neighbors=20, radius=1.0):
    # ---------- taipu_main_side. 预处理 ----------
    points = np.asarray(points, dtype=np.float64)
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    points = np.unique(points, axis=0)
    if points.shape[0] == 0:
        raise ValueError("输入点云为空或全为 NaN/Inf！")
    # ---------- 2. cKD-Tree ----------
    kdtree = cKDTree(points)
    # ---------- 3. 粗糙度（新：点到局部平面距离的标准差） ----------
    n_points = points.shape[0]
    roughness = np.zeros(n_points)
    # 查询邻域点
    distances, indices = kdtree.query(points, k=k_neighbors)
    # 计算每个点的局部粗糙度
    for i in range(n_points):
        neigh = points[indices[i]]  # K×3
        centroid = neigh.mean(axis=0)  # 3
        centered_neigh = neigh - centroid
        U, S, Vt = np.linalg.svd(centered_neigh)
        normal = Vt[-1]  # 最小特征值对应的主成分

        # 点到平面距离 = |(p - p0)·n|
        distances = np.abs(np.dot(centered_neigh, normal))
        roughness[i] = np.std(distances)
    # ---------- 4. 熵 ----------
    total = roughness.sum()
    if total == 0:
        return 0.0
    probs = roughness / total
    probs = probs[probs > 0]
    roughness_entropy = entropy(probs, base=2)

    return roughness_entropy


def apply_rotation(points, center_points, density, points_thresh, density_thresh, mode=None):
    if mode == "bridge":
        if len(points) < points_thresh:
            xoy_points = center_points.copy()
        else:
            xoy_points = points.copy()
    else:
        xoy_points = points.copy()
    # 将点云投射到xoy平面上
    xoy_points[:, 2] = 0
    unique_xoy_points = np.unique(xoy_points, axis=0)

    if len(points) < points_thresh and density < density_thresh:  # 15000有可能也是一半船
        # 将船按几何中心分成两部分
        min_bound = unique_xoy_points.min(axis=0)
        max_bound = unique_xoy_points.max(axis=0)
        xoy_center = (min_bound + max_bound) / 2
        xoy_lower = unique_xoy_points[unique_xoy_points[:, 0] < xoy_center[0]]
        xoy_higher = unique_xoy_points[unique_xoy_points[:, 0] >= xoy_center[0]]
        width_lower = xoy_lower[:, 1].max() - xoy_lower[:, 1].min()
        width_higher = xoy_higher[:, 1].max() - xoy_higher[:, 1].min()
        if width_lower < width_higher:
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
        # 如果是最后一个组，将余数加到最后一个组中
        if i == n - 1:
            end_index = total_points  # 最后一个组包含所有剩余的点
        else:
            end_index = start_index + group_size

        group = sorted_points[start_index:end_index]
        groups.append(group)
        start_index = end_index

    return groups


def find_most_frequent_value(arr, precision=1, n=2, threshold=0.1):
    # 将数组中的每个元素保留到指定的小数位数
    rounded_arr = np.round(arr, precision)

    # 对四舍五入后的值进行排序
    sorted_values = np.sort(rounded_arr)

    # 合并相近的值（前后相差不到 0.taipu_main_side）
    groups = []
    current_group = []

    for value in sorted_values:
        if not current_group:
            current_group.append(value)
        else:
            # 检查当前值与组内所有值的误差是否都在 0.taipu_main_side 以内
            if all(abs(value - v) <= threshold for v in current_group):
                current_group.append(value)
            else:
                groups.append(current_group)
                current_group = [value]

    if current_group:
        groups.append(current_group)

    # 找到最大的组
    largest_group = max(groups, key=len)

    # 最大组占比不到 20 % 或绝对个数不到 3 个，都认为“不集中”
    if len(largest_group) < max(3, n // 5):
        return None

    # 找到所有与众数相对应的原始值
    mask = np.isin(np.round(arr, precision), largest_group)
    corresponding_values = arr[mask]

    # 计算这些值的平均数
    mean_value = np.mean(corresponding_values) if len(corresponding_values) > 0 else None

    return mean_value


def filter_by_y(points, total_points, mode=None, save_ratio=3.0):
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
    y_total = total_points[:, 1]
    y_min, y_max = y_total.min(), y_total.max()
    y_center = (y_min + y_max) / 2.0
    width = (y_max - y_min) / 2

    # 计算 y - y_center
    y_diff = y - y_center
    mask = np.abs(y_diff) > (width / save_ratio)

    if mode == 'bank':  # 岸基模式
        mask_1 = mask & (y_diff < 0)
        return points[mask_1], 1
    elif mode == 'bridge':  # 桥基模式
        pos_cnt = np.sum(y > 0)
        if pos_cnt > 0.75 * len(y):  # 大部分点在 y 轴正方向
            mask_1 = mask & (y_diff < 0)  # 靠近坐标轴的一侧
            mask_2 = mask & (y_diff > 0)  # 远离坐标轴的一侧
            if abs(np.sum(mask_1) - np.sum(mask_2)) > 200:
                process_points, direction = (points[mask_2], -1) if np.sum(mask_1) < np.sum(mask_2) else (
                    points[mask_1], 1)
            else:
                process_points, direction = points[mask_1], 1
        else:  # 大部分点在 y 轴负方向
            mask_1 = mask & (y_diff > 0)  # 靠近坐标轴的一侧
            # mask_2 = mask & (y_diff < 0)  # 远离坐标轴的一侧
            # process_points, direction = (points[mask_2], 1) if np.sum(mask_1) < np.sum(mask_2) else (points[mask_1], -1)
            process_points, direction = (points[mask_1], -1)
        return process_points, direction
    else:
        return points[mask], 1


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


def detect_boundaries(points, radius=0.1, k=16, angle_threshold=np.pi * 0.7):
    """
    检测边界点
    :param points: 点云数据，形状为 (N, 3)
    :param k: 每个点的最近邻点数量
    :param angle_threshold: 角度阈值
    :return: 边界点
    """
    # 初始化边界点标记数组
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    boundary_points = np.zeros(len(points), dtype=bool)
    # 计算法向量
    normals = estimate_normals(points, radius=radius, max_nn=k)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.orient_normals_consistent_tangent_plane(k=k)
    normals = np.asarray(pcd.normals)
    # 遍历每个点，检查其是否为边界点
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(points[i], k)  # 查找16个最近邻点
        neighbor_normals = normals[idx]

        # 计算当前点与邻居点的法向量夹角
        angles = np.arccos(np.clip(np.dot(normals[i], neighbor_normals.T), -1.0, 1.0))
        if np.any(angles > angle_threshold):  # 角度大于0.6π的点被认为是边界点
            boundary_points[i] = True

    return points[~boundary_points]


def mahalanobis_filter_vec(points, nb_neighbors=16, std_ratio=2.0):
    """
    针对甲板平面优化的马氏距离滤波（向量化实现）。

    Parameters
    ----------
    points : np.ndarray
        输入点云，形状 (N, 3)。
    nb_neighbors : int
        每个点的邻居数量（不含自身）。
    std_ratio : float
        阈值倍数，inlier < mean + std_ratio * std。

    Returns
    -------
    inlier_pts : np.ndarray
        滤波后的点云，形状 (K, 3)。
    mask : np.ndarray
        保留掩码，形状 (N,)，bool。
    """
    tree = cKDTree(points)
    n = points.shape[0]

    # 一次性找邻居 (N, nb_neighbors)
    dists, idx = tree.query(points, k=nb_neighbors + 1)
    idx = idx[:, 1:]  # 去掉自身
    neigh = points[idx]  # (N, nb_neighbors, 3)

    # 邻居均值 (N, 3)
    μ = neigh.mean(axis=1)  # (N, 3)

    # 邻居中心化 (N, nb_neighbors, 3)
    diff = neigh - μ[:, None, :]  # (N, nb_neighbors, 3)

    # 批量协方差 (N, 3, 3)
    cov = np.einsum('nki,nkj->nij', diff, diff) / nb_neighbors
    cov += 1e-6 * np.eye(3)  # 正则化

    # 批量逆矩阵 (N, 3, 3)
    inv_cov = np.linalg.inv(cov)

    # 自身到邻居均值的残差 (N, 3)
    self_diff = points - μ  # (N, 3)

    # 马氏距离平方 (N,)
    md2 = np.einsum('ni,nij,nj->n', self_diff, inv_cov, self_diff)
    md = np.sqrt(md2)

    # 阈值判定
    mean_md = md.mean()
    std_md = md.std()
    mask = md < (mean_md + std_ratio * std_md)

    return points[mask], mask


def apply_rotation_matrix_to_point_cloud(points, R):
    """
    对 (N,3) 点云应用 3×3 旋转 或 4×4 齐次矩阵
    返回 (N,3)
    """
    points = np.asarray(points, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)

    if R.shape == (3, 3):
        return (R @ points.T).T
    elif R.shape == (4, 4):
        # 升维 → 齐次坐标
        h_pts = np.hstack([points, np.ones((points.shape[0], 1))])
        h_rot = (R @ h_pts.T).T
        return h_rot[:, :3]          # 降回 3D
    else:
        raise ValueError("R 必须是 (3,3) 或 (4,4)")

# def compute_curvature(points, k=30):
#     """
#     用 cKDTree 计算逐点曲率，返回与 points 同长度的 ndarray
#     """
#     tree = cKDTree(points)
#     curv = np.zeros(len(points))
#
#     # 一次性查询 k+taipu_main_side 邻域（第 0 个是自己）
#     dists, idxs = tree.query(points, k=k+taipu_main_side, workers=-taipu_main_side)   # workers=-taipu_main_side 自动并行
#     for i, idx in enumerate(idxs):
#         neigh = points[idx[taipu_main_side:]]          # 去掉自己
#         cov = np.cov(neigh.T)
#         eigvals = np.linalg.eigvalsh(cov)
#         curv[i] = eigvals[0] / (eigvals.sum() + 1e-12)
#
#     return curv
#
# def get_global_index(divide_rotation_center, all_points):
#     """把 divide_rotation_center 的局部索引映射到 self.all_points 的全局索引"""
#     # 示例：如果 divide_rotation_center 就是 all_points 的切片，可直接返回
#     return np.arange(len(all_points))[np.isin(all_points, divide_rotation_center).all(axis=taipu_main_side)]


class LidarFunc(object):
    def __init__(self, parser_dict, **kwargs):
        self.parser_dict = parser_dict
        self.extra_params = kwargs

    def freeboard_calculation(self, points, water_level=None):
        # points = apply_rotation_matrix_to_point_cloud(points, np.asarray(
        #     [[0.9999936784236516, 4.441341249212944e-07, 0.00355571547472536],
        #      [4.441341249212944e-07, 0.9999999687965295, -0.0002498134158006251],
        #      [-0.00355571547472536, 0.0002498134158006251, 0.9999936472201811]]))
        # points = points[points[:, 2] > -11.68]
        o3d.utility.random.seed((42))
        start_time = time.time()
        density_all = compute_average_density(points, k_neighbors=16)
        points = voxel_filter(points, voxel_size=self.parser_dict["voxel_size"] * density_all)
        print(
            f"Voxel filter 耗时: {time.time() - start_time:.6f} 秒, 体素大小：{self.parser_dict['voxel_size'] * density_all}")
        x_min = points[:, 0].min()
        x_max = points[:, 0].max()
        x_center = (x_max + x_min) / 2
        print("船舶中心x值：", x_center)
        if x_center < self.parser_dict['x_min'] or x_center > self.parser_dict['x_max']:
            return np.nan, np.nan, np.nan
        start_time = time.time()
        center_points = select_center(points)
        print(f"select_center 耗时: {time.time() - start_time:.6f} 秒")
        start_time = time.time()
        density = compute_average_density(center_points, k_neighbors=16)
        print(f"compute_average_density 耗时: {time.time() - start_time:.6f} 秒")
        start_time = time.time()
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
        ) = apply_rotation(points, center_points, density,
                           points_thresh=self.parser_dict['points_thresh'],
                           density_thresh=self.parser_dict['density_thresh'],
                           mode=self.parser_dict['func_mode'])
        print(f"apply_rotation 耗时: {time.time() - start_time:.6f} 秒")
        # 将rotation_center_points按x值分成n等分
        if self.parser_dict['func_mode'] == "bridge":
            boundary_points = detect_boundaries(rotation_center_points, radius=0.5 * density, k=16,
                                                angle_threshold=np.pi * 0.7)
            process_points, direction = filter_by_y(boundary_points, rotation_total_points,
                                                    mode=self.parser_dict['func_mode'])
            print("筛选结果", len(process_points), direction)
        elif self.parser_dict['func_mode'] == "bank":
            start_time = time.time()
            process_points, direction = filter_by_y(rotation_center_points, rotation_total_points,
                                                    mode=self.parser_dict['func_mode'],
                                                    save_ratio=self.parser_dict['save_ratio'])
        print(f"filter_by_y 耗时: {time.time() - start_time:.6f} 秒")
        print("船舶边缘点数：", len(process_points), "计算方向", direction)
        if len(process_points) > self.parser_dict['filter_thresh']:
            nb_neighbors = self.parser_dict['nb_neighbors'] * density
            std_ratio = self.parser_dict['std_ratio'] * density
            start_time = time.time()
            process_points, _ = mahalanobis_filter_vec(process_points, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            print(f"mahalanobis_filter_vec 耗时: {time.time() - start_time:.6f} 秒")
        n = max(1, int(len(process_points) / self.parser_dict['divide_thresh']))
        print("分段个数：", n)
        start_time = time.time()
        divide_rotation_center_list = divide_points(process_points, axis=0, n=n)
        print(f"divide_points 耗时: {time.time() - start_time:.6f} 秒")
        freeboard_upper_points = []
        start_time = time.time()
        for divide_rotation_center in divide_rotation_center_list:
            # 对每个部分提取y坐标
            y_coords = divide_rotation_center[:, 1] + self.parser_dict['axis_offset']
            # taipu_main_side. 计算排序索引（升序）
            score = direction * np.abs(y_coords)
            order = np.argsort(score)
            # 2. 逐步扩大窗口，直到方差超标
            for k in range(1, len(order) + 1):
                idx_win = order[:k]
                var = np.var(y_coords[idx_win])
                if var >= self.parser_dict['var_thresh']:
                    # 超标了，取上一次的结果
                    selected = order[:k - 1]
                    break
            else:
                # 整个数组方差都不超标，全取
                selected = order
            y_top_index = selected
            # 对其z轴大小排序
            max_z_idx_in_top = np.argmax(divide_rotation_center[y_top_index][:, 2])
            global_max_z_idx = y_top_index[max_z_idx_in_top]
            freeboard_upper_point = divide_rotation_center[global_max_z_idx]  # 获取该点的 z 坐标作为干舷的上界
            freeboard_upper_points.append(freeboard_upper_point)
        print(f"divide_points_select 耗时: {time.time() - start_time:.6f} 秒")
        # 拿出freeboard_upper_points保留两位小数后频数最多的点作为freeboard_upper_point
        rotation_center_points = np.vstack([rotation_center_points, freeboard_upper_points])
        process_points = np.vstack([process_points, freeboard_upper_points])
        freeboard_upper_points = np.asarray(freeboard_upper_points)
        start_time = time.time()
        freeboard_upper_points_rotation = reverse_rotation(freeboard_upper_points, rotation_matrix, center)
        print(f"reverse_rotation 耗时: {time.time() - start_time:.6f} 秒")
        start_time = time.time()
        freeboard_upper_bound = find_most_frequent_value(freeboard_upper_points_rotation[:, 2], n=n)
        center_points = np.vstack([center_points, freeboard_upper_points_rotation])
        print(f"find_most_frequent_value 耗时: {time.time() - start_time:.6f} 秒")
        if freeboard_upper_bound is None:
            freeboard_upper_bound = np.median(freeboard_upper_points[:, 2])
        if water_level is not None:
            freeboard_lower_bound = water_level
            deck = abs(freeboard_upper_bound - freeboard_lower_bound)
        else:
            freeboard_lower_bound = center_points[:, 2].min()
            deck = freeboard_upper_bound - freeboard_lower_bound
        return (
            points,
            center_points,
            density,
            principal_direction,
            xoy_points,
            center,
            rotation_total_points,
            rotation_center_points,
            process_points,
            divide_rotation_center_list,
            freeboard_upper_bound,
            freeboard_lower_bound,
            deck,
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


def vertical_neighbor_stats(points):
    """
    返回竖直相邻点对的最小、最大、平均 Δz
    """
    # 按 x 值分组
    unique_x = np.unique(points[:, 0])
    all_dz = []

    for x in unique_x:
        # 提取当前 x 值的所有点
        group_points = points[points[:, 0] == x]
        # 按 z 值升序排序
        z_sorted = np.sort(group_points[:, 2])
        # 计算相邻点的 Δz
        dz = np.diff(z_sorted)
        # 将当前分组的 Δz 添加到总列表中
        all_dz.extend(dz)

    # 将所有 Δz 转换为 NumPy 数组
    all_dz = np.array(all_dz)

    # 计算最小、最大和平均 Δz
    min_dz = np.min(all_dz)
    max_dz = np.max(all_dz)
    mean_dz = np.mean(all_dz)

    print("竖直相邻最小距离：", float(min_dz))
    print("竖直相邻最大距离：", float(max_dz))
    print("竖直相邻平均距离：", float(mean_dz))

    return min_dz, max_dz, mean_dz


def make_xoy_plane_through_point(point, size=50.0, color=[0.3, 0.7, 0.9]):
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
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
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
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def xyz_to_standard(pcd, inverse=False):
    """
    把『x右 y下 z里』→『x右 y前 z上』
    inverse=True 则反向。
    """
    T = np.array([[0, -1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]], dtype=float)
    if inverse:
        T = T.T  # 逆矩阵就是转置
    return pcd.transform(T)


def _box_corners_2d(max_x, max_y, min_x, min_y, z=0.0):
    """返回方形在 z=z 平面上的 4 个角点 (4,3)。顺序：↘↗↖↙"""
    return np.array([
        [max_x, min_y, z],  # 右下
        [max_x, max_y, z],  # 右上
        [min_x, max_y, z],  # 左上
        [min_x, min_y, z]  # 左下
    ])


def filter_area(pcd, areas, z=0.0, vis=False):
    """
    用若干 2D 方形区域过滤点云，并生成区域边框线。

    参数
    ----
    pcd : o3d.geometry.PointCloud
        输入点云
    areas : list of tuple
        (max_x, max_y, min_x, min_y)
    z : float, optional
        画线框时所在的 z 平面高度（默认 0）

    返回
    ----
    inside_pcd  : o3d.geometry.PointCloud
    outside_pcd : o3d.geometry.PointCloud
    line_set    : o3d.geometry.LineSet
    """
    xyz = np.asarray(pcd.points)  # (N,3)
    N = xyz.shape[0]

    # taipu_main_side. 过滤
    mask = np.zeros(N, dtype=bool)
    for max_x, max_y, min_x, min_y in areas:
        in_x = (xyz[:, 0] >= min_x) & (xyz[:, 0] <= max_x)
        in_y = (xyz[:, 1] >= min_y) & (xyz[:, 1] <= max_y)
        mask |= (in_x & in_y)

    inside_pcd = pcd.select_by_index(np.where(mask)[0])
    outside_pcd = pcd.select_by_index(np.where(~mask)[0])

    # 2. 构造 LineSet
    points = []  # 所有方框的顶点
    lines = []  # 每框四条边
    idx_offset = 0
    for max_x, max_y, min_x, min_y in areas:
        corners = _box_corners_2d(max_x, max_y, min_x, min_y, z)
        points.append(corners)
        # 四条线段：0-taipu_main_side, taipu_main_side-2, 2-3, 3-0
        lines.append(
            np.array([[0, 1], [1, 2], [2, 3], [3, 0]]) + idx_offset
        )
        idx_offset += 4

    points = np.vstack(points)
    lines = np.vstack(lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if vis:
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer(
            "area",
            width=1660, height=900)
        vis.show_skybox(False)
        vis.show_settings = True
        # 设置背景颜色为黑色
        background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
        vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None
        vis.add_geometry("inside_pcd", inside_pcd)
        print(np.asarray(inside_pcd.points)[:, 1].max())
        print(np.asarray(inside_pcd.points)[:, 1].min())
        vis.add_geometry("outside_pcd", outside_pcd)
        vis.add_geometry("line_set", line_set)
        app.add_window(vis)
        app.run()

    return outside_pcd


if __name__ == '__main__':
    # file_path = None
    file_path = rf"D:\program\li3D-ML\data\Taipu-main\08_10_37_39_ced36c26-45ae-445b-8a08-5e16a8628b22_1\detail\deck_pcd\ship_1757299097.5516424_nan_105695_-9.441085596937105.pcd"
    o3d.utility.random.seed(42)
    # dataset = 'error'
    dataset = 'Taipu-main'
    directory_path = f"../data/{dataset}"
    # 加载算法参数文件
    with open('json_files/parser_dict_bank.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    parser_dict = data.get('parser_dict', {})
    # 初始化雷达方法
    lidar_func = LidarFunc(parser_dict)
    # 获取目标路径下的所有文件和目录
    entries = os.listdir(directory_path)
    # 筛选出第一级目录
    dirs = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]
    d = 0
    for dir in dirs[d:]:
        # 构建当前目录的完整路径
        frame_freeboard_upper_bound = []
        pcd_files = glob.glob(f'{directory_path}/{dir}/*.pcd', recursive=True)
        n = 0
        for i, pcd_file in enumerate(pcd_files[n:], start=n):
            s = time.time()
            if file_path is not None:
                pcd = o3d.io.read_point_cloud(file_path, format="pcd")
                l = split_filename_from_path(file_path)
            else:
                l = split_filename_from_path(pcd_file)
                pcd = o3d.io.read_point_cloud(pcd_file, format="pcd")
            # 其他过滤
            # pcd = xyz_to_standard(pcd)
            # points = np.asarray(pcd.points)
            # filtered_points = points[points[:, 2] > -11.68]
            # pcd.points = o3d.utility.Vector3dVector(filtered_points)
            # pcd = filter_area(pcd, areas = [(300, 350, -300, 250)], vis=True)
            # pcd = filter_area(pcd, areas = [(300, 85, -300, 23)], vis=True)
            # 文件名过滤
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:int(l[-2])])
            # start_time = time.time()
            process_points = np.asarray(pcd.points).copy()
            # process_points = statistical_outlier(process_points, nb_neighbors=2, std_ratio=10)
            # print("滤波时间：", time.time() - start_time)
            # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
            (
                points,
                center_points,
                density,
                principal_direction,
                xoy_points,
                center,
                rotation_total_points,
                rotation_center_points,
                filter_rotation_center_points,
                divide_rotation_center_list,
                freeboard_upper_bound,
                freeboard_lower_bound,
                deck,
                n,
                freeboard_upper_points,
                vals,
                vecs_T,
                selected_part,
            ) = lidar_func.freeboard_calculation(np.asarray(process_points))
            print(f"运行时间：{time.time() - s}")
            # vertical_neighbor_stats(rotation_total_points)
            # 绘制折线图
            plt.plot(freeboard_upper_points[:, 2], marker='o', linestyle='-', label='Z Coordinates')  # 使用圆形标记和实线
            if freeboard_upper_bound is not None:
                plt.axhline(y=freeboard_upper_bound, color='r', linestyle='--',
                            label=f'Platform Average: {freeboard_upper_bound:.2f}')
            plt.title('Z Coordinates of Selected Points')
            plt.xlabel('Index')
            plt.ylabel('Z Coordinate')
            plt.grid(True)  # 添加网格线
            plt.legend()
            plt.show()

            # 构建open3d对象
            center_pcd = points2pcd(center_points)
            center_pcd.paint_uniform_color([1, 0, 0])
            colors = center_pcd.colors
            for i in range(1, n + 1):
                colors[-i] = [0, 1, 0]
            center_pcd.colors = o3d.utility.Vector3dVector(colors)

            xoy_pcd = points2pcd(xoy_points)
            selected_pcd = points2pcd(selected_part)

            print("主方向向量和x轴的夹角：", angle_with_x_axis(principal_direction))

            # 创建线段的起点和终点
            line_set = []
            start_point = (xoy_points.min(axis=0) + xoy_points.max(axis=0)) / 2
            for direction, value in zip(vecs_T, vals):
                if value == 0.0:
                    continue
                line_set.append(build_line(start_point, direction, value))

            box = build_box(xoy_points)

            pcd_process = points2pcd(points)

            # 点云投影代码（没用）
            # 创建新的点云坐标数组，并将所有点的 x 坐标设置为 0
            rotation_total_pcd = points2pcd(rotation_total_points)

            rotation_center_pcd = points2pcd(rotation_center_points)
            rotation_center_pcd.paint_uniform_color([0, 0, 1])
            colors = rotation_center_pcd.colors
            for i in range(1, n + 1):
                colors[-i] = [0, 1, 0]
            rotation_center_pcd.colors = o3d.utility.Vector3dVector(colors)

            filter_rotation_center_pcd = points2pcd(filter_rotation_center_points)
            filter_rotation_center_pcd.paint_uniform_color([1, 0, 0])
            colors = filter_rotation_center_pcd.colors
            for i in range(1, n + 1):
                colors[-i] = [0, 1, 0]
            filter_rotation_center_pcd.colors = o3d.utility.Vector3dVector(colors)

            divide_rotation_center_pcds = []
            for divide_rotation_center_points in divide_rotation_center_list:
                divide_rotation_center_pcd = points2pcd(divide_rotation_center_points)
                divide_rotation_center_pcds.append(divide_rotation_center_pcd)

            new_points = np.asarray(rotation_center_pcd.points).copy()
            new_points[:, 0] = 0

            # 创建新的点云对象
            new_pcd = points2pcd(new_points)
            new_pcd.colors = rotation_center_pcd.colors

            platform_x = make_yoz_plane_through_point(center)
            center[2] = freeboard_upper_bound
            platform = make_xoy_plane_through_point(center, size=20.0, color=[1, 1, 0])
            # center[2] = -7.93
            # platform_z = make_xoy_plane_through_point(center, size=20.0)

            # ui（没用）
            app = gui.Application.instance
            app.initialize()
            vis = o3d.visualization.O3DVisualizer(
                f"第{i}帧点云 - {pcd_file if file_path is None else os.path.basename(file_path)} - density：{density} 上边界高度为：{freeboard_upper_bound}, 下边界高度为：{freeboard_lower_bound}, 干舷高度为：{deck}",
                width=1660, height=900)
            vis.show_skybox(False)
            vis.show_settings = True
            # 设置背景颜色为黑色
            background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
            vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None

            vis.add_geometry("pcd", pcd)
            vis.add_geometry("pcd_process", pcd_process)
            vis.add_geometry("center_pcd", center_pcd)
            vis.add_geometry("xoy_pcd", xoy_pcd)
            vis.add_geometry("selected_pcd", selected_pcd)
            vis.add_geometry("box", box)
            for i, line in enumerate(line_set):
                vis.add_geometry(f"line_{i}", line)
            vis.add_geometry("rotation_total_pcd", rotation_total_pcd)
            vis.add_geometry("rotation_center_pcd", rotation_center_pcd)
            vis.add_geometry("filter_rotation_center_pcd", filter_rotation_center_pcd)
            # for i, pcd in enumerate(divide_rotation_center_pcds):
            #     vis.add_geometry(f"divide_pcd{i}", pcd)
            #     print(f"divide_pcd{i}点数：", len(pcd.points), "长度：",
            #           np.asarray(pcd.points)[:, 0].max() - np.asarray(pcd.points)[:, 0].min())
            vis.add_geometry("new_pcd", new_pcd)
            vis.add_geometry("platform", platform)
            # vis.add_geometry("platform_z", platform_z)

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
