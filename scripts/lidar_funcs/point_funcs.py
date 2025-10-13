import numpy as np
import open3d as o3d
import os
import open3d.visualization.gui as gui
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import minimize
from scipy.stats import entropy
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    点云选取几何中心附近的点。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :param return_index: 是否返回索引，默认为 False。
    :type return_index: bool
    :return: 几何中心附近的点，可选返回索引。
    :rtype: np.ndarray, np.ndarray (可选)
    """
    xoy_points = points.copy()
    xoy_points[:, 2] = 0
    min_bound = xoy_points.min(axis=0)
    max_bound = xoy_points.max(axis=0)
    range_ = abs(max_bound[0] - min_bound[0]) / 6
    point_center = (min_bound + max_bound) / 2
    x_filter_indices = \
    np.where((points[:, 0] >= (point_center[0] - range_)) & (points[:, 0] <= (point_center[0] + range_)))[0]
    process_points = points[x_filter_indices]
    if return_index:
        return process_points, x_filter_indices
    else:
        return process_points


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

    # taipu_main_side. 一次性找邻居 (N, nb_neighbors)
    dists, idx = tree.query(points, k=nb_neighbors + 1)
    neigh = points[idx[:, 1:]]  # (N, nb_neighbors, 3)

    # 2. 邻居均值 (N, 3)
    μ = neigh.mean(axis=1, keepdims=True)  # (N, taipu_main_side, 3)

    # 3. 邻居中心化 (N, nb_neighbors, 3)
    diff = neigh - μ  # (N, nb_neighbors, 3)

    # 4. 批量协方差 (N, 3, 3)
    cov = np.einsum('nki,nkj->nij', diff, diff) / (nb_neighbors - 1 + 1e-12)
    cov += 1e-6 * np.eye(3)[None, :, :]  # 正则化

    # 5. 批量逆矩阵 (N, 3, 3)
    inv_cov = np.linalg.inv(cov)

    # 6. 自身到邻居均值的残差 (N, 3)
    self_diff = points - μ.squeeze(1)  # (N, 3)

    # 7. 马氏距离平方 (N,)
    md2 = np.einsum('ni,nij,nj->n', self_diff, inv_cov, self_diff)
    md = np.sqrt(md2)

    # 8. 阈值判定
    mean_md = md.mean()
    std_md = md.std()
    mask = md < (mean_md + std_ratio * std_md)

    return points[mask], mask


def statistical_outlier(points, nb_neighbors=16, std_ratio=1.0):
    """
    点云统计滤波。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :param nb_neighbors: 邻近点数，默认为 16。
    :type nb_neighbors: int
    :param std_ratio: 标准差比率，默认为 taipu_main_side.0。
    :type std_ratio: float
    :return: 滤波后的点云。
    :rtype: np.ndarray
    """
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors)
    mean_distances = np.mean(distances, axis=1)
    mean_distance = np.mean(mean_distances)
    stddev = np.std(mean_distances)
    inliers = np.abs(mean_distances - mean_distance) < std_ratio * stddev
    return points[inliers]


def compute_principal_direction(points):
    """
    点云主方向向量计算。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :return: 主方向向量，特征值，特征向量矩阵。
    :rtype: np.ndarray, np.ndarray, np.ndarray
    """
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    principal = vecs[:, np.argmax(vals)]
    if np.dot(principal, [1, 0, 0]) < 0:
        principal = -principal
    vecs_T = vecs.T
    for i in range(vecs_T.shape[0]):
        if np.dot(vecs_T[i], [1, 0, 0]) < 0:
            vecs_T[i] = -vecs_T[i]
    return principal, vals, vecs_T


def build_rotation_matrix(principal_direction):
    """
    构建将主方向对齐到 x 轴的旋转矩阵，仅绕 y 轴和 z 轴旋转。

    :param principal_direction: 主方向向量。
    :type principal_direction: np.ndarray
    :return: 旋转矩阵。
    :rtype: np.ndarray
    """
    principal_direction = principal_direction / np.linalg.norm(principal_direction)
    angle_y = np.arctan2(principal_direction[2], principal_direction[0])
    rotation_matrix_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    rotated_principal_direction = np.dot(rotation_matrix_y, principal_direction)
    angle_z = np.arctan2(rotated_principal_direction[1], rotated_principal_direction[0])
    rotation_matrix_z = np.array([
        [np.cos(angle_z), np.sin(angle_z), 0],
        [-np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    rotation_matrix = np.dot(rotation_matrix_z, rotation_matrix_y)
    return rotation_matrix


def compute_average_density(points):
    """
    点云计算体密度。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :return: 点云的平均密度。
    :rtype: float
    """
    if points.size == 0:
        return 0.0
    min_b, max_b = points.min(axis=0), points.max(axis=0)
    vol = np.prod(max_b - min_b)
    return len(points) / vol if vol > 1e-12 else float('inf')


def estimate_normals(points: np.ndarray, radius: float, max_nn: int) -> np.ndarray:
    """
    向量化 PCA 法向量估计（CPU）。

    :param points: 输入点云，形状为 (N, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :param radius: 邻域搜索半径。
    :type radius: float
    :param max_nn: 每个点的最大邻居数（≥ 3）。
    :type max_nn: int
    :return: 估计的法向量，形状为 (N, 3) 的 NumPy 数组，已单位化；邻居不足或退化时置零。
    :rtype: np.ndarray
    """
    N = points.shape[0]
    tree = cKDTree(points)

    # taipu_main_side. 邻居查询并截断
    idx_lists = tree.query_ball_point(points, r=radius, p=2, workers=-1)
    idx_lists = [lst[:max_nn] for lst in idx_lists]
    lens = np.array([len(lst) for lst in idx_lists], dtype=np.int64)

    mask = lens >= 3
    if not mask.any():
        return np.zeros_like(points)

    max_k = max_nn
    neigh_idx = np.full((N, max_k), -1, dtype=np.int64)
    neigh_msk = np.arange(max_k) < lens[:, None]

    for i, lst in enumerate(idx_lists):
        k = len(lst)
        if k:
            neigh_idx[i, :k] = lst

    neigh = points[neigh_idx]
    neigh = np.where(neigh_msk[..., None], neigh, np.nan)

    means = np.nanmean(neigh, axis=1, keepdims=True)
    centered = neigh - means

    valid_cnt = lens.astype(points.dtype)[:, None, None]
    cov = np.einsum('nki,nkj->nij', centered, centered) / (valid_cnt - 1 + 1e-12)

    cov += np.eye(3, dtype=points.dtype) * 1e-6

    normals = np.zeros_like(points)
    try:
        _, _, Vt = np.linalg.svd(cov)
        normals[mask] = Vt[mask, -1, :]
    except np.linalg.LinAlgError:
        pass  # 退化点保持零向量

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norms + 1e-12
    return normals


def compute_roughness_entropy(points, k_neighbors=8, radius=1.0):
    """
    基于局部粗糙度分布的香农熵计算。

    对每个点取其 k 近邻，拟合局部切平面，计算邻域点到该平面的距离标准差作为
    局部粗糙度；最后对所有点的粗糙度分布求香农熵，衡量整体不规则程度。

    :param points: 点云数据，形状为 (N, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :param k_neighbors: 每个点用于拟合平面的邻居数量，默认为 8。
    :type k_neighbors: int
    :param radius: 搜索邻居时允许的最大半径（当前未启用，保留接口）。
    :type radius: float
    :return: 粗糙度分布的香农熵，以 2 为底，单位为 bit。
    :rtype: float
    """
    # ---------- taipu_main_side. 预处理 ----------
    points = np.asarray(points, dtype=np.float64)
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    points = np.unique(points, axis=0)
    if points.shape[0] == 0:
        raise ValueError("输入点云为空或全为 NaN/Inf！")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # ---------- 2. KD-Tree ----------
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # ---------- 3. 粗糙度（新：点到局部平面距离的标准差） ----------
    roughness = np.zeros(len(pcd.points))
    pca = PCA(n_components=3)

    for i in range(len(pcd.points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k_neighbors)
        if len(idx) < 3:
            continue

        neigh = points[idx]  # K×3
        centroid = neigh.mean(axis=0)  # 3
        pca.fit(neigh - centroid)  # 去中心化
        normal = pca.components_[-1]  # 最小特征值 → 法向量

        # 点到平面距离 = |(p - p0)·n|
        distances = np.abs(np.dot(neigh - centroid, normal))
        roughness[i] = np.std(distances)

    # ---------- 4. 熵 ----------
    total = roughness.sum()
    if total == 0:
        return 0.0
    probs = roughness / total
    probs = probs[probs > 0]
    roughness_entropy = entropy(probs, base=2)
    return roughness_entropy


def select_largest_connected_component(points, radius=1.5):
    """
    从点云中选择最大的连通域。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :param radius: 邻域搜索的半径，默认为 taipu_main_side.5。
    :type radius: float
    :return: 最大的连通域的点云。
    :rtype: np.ndarray
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    num_points = len(points)
    adj_matrix = np.zeros((num_points, num_points), dtype=int)
    for i in range(num_points):
        _, indices, _ = kdtree.search_radius_vector_3d(points[i], radius)
        for j in indices:
            if i != j:
                adj_matrix[i, j] = 1
    graph = csr_matrix(adj_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_label = unique_labels[np.argmax(counts)]
    largest_indices = np.where(labels == largest_label)[0]
    return points[largest_indices]


def apply_rotation(points, center_points, density):
    """
    应用旋转矩阵到点云。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :param center_points: 中心点数据，形状为 (n, 3) 的 NumPy 数组。
    :type center_points: np.ndarray
    :param density: 点云密度。
    :type density: float
    :return: 旋转后的点云，主方向向量，原始点云，旋转矩阵，中心点，特征值，特征向量矩阵，选定部分。
    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    xoy_points = points.copy()
    xoy_points[:, 2] = 0
    unique_xoy_points = np.unique(xoy_points, axis=0)
    if len(points) < 15000 and density < 0.8:
        min_bound = unique_xoy_points.min(axis=0)
        max_bound = unique_xoy_points.max(axis=0)
        xoy_center = (min_bound + max_bound) / 2
        xoy_lower = unique_xoy_points[unique_xoy_points[:, 0] < xoy_center[0]]
        xoy_higher = unique_xoy_points[unique_xoy_points[:, 0] >= xoy_center[0]]
        density_lower = compute_average_density(xoy_lower)
        density_higher = compute_average_density(xoy_higher)
        if density_lower < density_higher:
            selected_part = xoy_lower
        else:
            selected_part = xoy_higher
    else:
        selected_part = xoy_points
    principal_direction, vals, vecs_T = compute_principal_direction(selected_part)
    rotation_matrix = build_rotation_matrix(principal_direction)
    center = np.mean(center_points, axis=0)
    rotated_points = np.dot(center_points - center, rotation_matrix.T) + center
    return rotated_points, principal_direction, xoy_points, rotation_matrix, center, vals, vecs_T, selected_part


def reverse_rotation(rotated_points, rotation_matrix, center):
    """
    将旋转后的点反变换回原始坐标系。

    :param rotated_points: 旋转后的点云，形状为 (n, 3) 的 NumPy 数组。
    :type rotated_points: np.ndarray
    :param rotation_matrix: 旋转矩阵，形状为 (3, 3) 的 NumPy 数组。
    :type rotation_matrix: np.ndarray
    :param center: 旋转中心，形状为 (3,) 的 NumPy 数组。
    :type center: np.ndarray
    :return: 反变换后的点云。
    :rtype: np.ndarray
    """
    shifted_points = rotated_points - center
    reversed_points = np.dot(shifted_points, rotation_matrix)
    original_points = reversed_points + center
    return original_points


def divide_points(points, axis=0, n=3):
    """
    根据指定轴坐标对点云进行排序，并将其分成 n 等分。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :param axis: 指定轴，默认为 0（x 轴）。
    :type axis: int
    :param n: 分组数，默认为 3。
    :type n: int
    :return: 分组后的点云列表。
    :rtype: list
    """
    sorted_indices = np.argsort(points[:, axis])
    sorted_points = points[sorted_indices]
    total_points = len(points)
    if total_points < n:
        raise ValueError(f"点云中点的数量不足以分成 {n} 等分")
    group_size = total_points // n
    groups = []
    start_index = 0
    for i in range(n):
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


def points2pcd(points):
    """
    将点云数据转换为 Open3D 的 PointCloud 对象。

    :param points: 点云数据，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :return: Open3D 的 PointCloud 对象。
    :rtype: o3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


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


def split_filename_from_path(file_path):
    """
    从完整路径中提取文件名（不包括扩展名），并按 '_' 分隔为多个部分。

    :param file_path: 完整的文件路径。
    :type file_path: str
    :return: 文件名的各个部分。
    :rtype: list
    """
    file_name_with_extension = os.path.basename(file_path)
    file_name, _ = os.path.splitext(file_name_with_extension)
    parts = file_name.split('_')
    parts = [part for part in parts if part]
    return parts


def find_first_platform_and_subsequent_averages(z_coordinates):
    """
    找到第一个显著变化的点，并计算后续平台的平均值。

    :param z_coordinates: z 坐标数组，形状为 (n,) 的 NumPy 数组。
    :type z_coordinates: np.ndarray
    :return: 第一个平台的平均值。
    :rtype: float
    """
    diff = np.diff(z_coordinates)
    threshold = np.mean(np.abs(diff)) * 0.5
    change_points = np.where(np.abs(diff) > threshold)[0]
    if change_points.size == 0:
        return None
    first_change_index = change_points[0] + 1
    platform_points = []
    platform_indices = []
    window_size = max(3, len(z_coordinates) // 10)
    i = first_change_index
    while i < len(z_coordinates):
        window = z_coordinates[i:i + window_size]
        if len(window) >= window_size and np.std(window) < threshold:
            platform_points.extend(window)
            platform_indices.extend(range(i, min(i + window_size, len(z_coordinates))))
            i += window_size
        else:
            i += 1
    if not platform_points:
        return None
    platform = np.mean(platform_points)
    return platform


def build_line(start, direction, value, color=None):
    """
    构建一条线段。

    :param start: 起点，形状为 (3,) 的 NumPy 数组。
    :type start: np.ndarray
    :param direction: 方向向量，形状为 (3,) 的 NumPy 数组。
    :type direction: np.ndarray
    :param value: 线段长度。
    :type value: float
    :param color: 线段颜色，默认为红色。
    :type color: list
    :return: Open3D 的 LineSet 对象。
    :rtype: o3d.geometry.LineSet
    """
    if color is None:
        color = [1, 0, 0]
    end = start + direction * value
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start, end])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])
    return line_set


def build_box(points):
    """
    构建并绘制三维点云的包围盒。

    :param points: 三维点云，形状为 (n, 3) 的 NumPy 数组。
    :type points: np.ndarray
    :return: Open3D 的 LineSet 对象。
    :rtype: o3d.geometry.LineSet
    """
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)
    vertices = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z]
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(vertices)
    box.lines = o3d.utility.Vector2iVector(edges)
    box.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(edges))])
    return box


def angle_with_x_axis(vector):
    """
    计算向量与 x 轴的夹角（以度为单位）。

    :param vector: 二维或三维向量，形状为 (2,) 或 (3,) 的 NumPy 数组。
    :type vector: np.ndarray
    :return: 向量与 x 轴的夹角（度）。
    :rtype: float
    """
    if len(vector) == 2:
        x, y = vector
        angle = np.arctan2(y, x)
    elif len(vector) == 3:
        x, y, z = vector
        angle = np.arctan2(y, x)
    else:
        raise ValueError("向量必须是二维或三维的")
    return np.degrees(angle)


def gaussian_3d(xyz, A, u_xyz, sigma):
    """
    三维高斯函数。

    :param xyz: (x, y, z) 坐标，形状为 (3,) 的 NumPy 数组。
    :type xyz: np.ndarray
    :param A: 高斯函数的振幅。
    :type A: float
    :param u_xyz: (u_x, u_y, u_z) 中心，形状为 (3,) 的 NumPy 数组。
    :type u_xyz: np.ndarray
    :param sigma: 标量宽度。
    :type sigma: float
    :return: 三维高斯函数的值。
    :rtype: float
    """
    diff = (xyz - u_xyz) / sigma
    return A * np.exp(-0.5 * np.dot(diff, diff))


def f_3d(xyz, A, u_list, sigma):
    """
    多个三维高斯函数的叠加。

    :param xyz: (x, y, z) 坐标，形状为 (3,) 的 NumPy 数组。
    :type xyz: np.ndarray
    :param A: 高斯函数的振幅。
    :type A: float
    :param u_list: 中心点列表，每个中心点为一个形状为 (3,) 的 NumPy 数组。
    :type u_list: list
    :param sigma: 标量宽度。
    :type sigma: float
    :return: 多个三维高斯函数叠加后的值。
    :rtype: float
    """
    return np.sum([gaussian_3d(xyz, A, u, sigma) for u in u_list])


def gaussian_approximation_3d(u_list, sigma=1.0, A=1.0, distinguishing_coefficient=1.0):
    """
    寻找叠加高斯函数的最大值点 (x*, y*, z*)。

    :param u_list: 中心点列表，每个中心点为一个形状为 (3,) 的 NumPy 数组。
    :type u_list: list
    :param sigma: 标量宽度，默认为 taipu_main_side.0。
    :type sigma: float
    :param A: 高斯函数的振幅，默认为 taipu_main_side.0。
    :type A: float
    :param distinguishing_coefficient: 区分系数，默认为 taipu_main_side.0。
    :type distinguishing_coefficient: float
    :return: 叠加高斯函数的最大值点 (x*, y*, z*)。
    :rtype: np.ndarray
    """
    # taipu_main_side. 构造候选网格
    u_arr = np.asarray(u_list)
    min_u, max_u = u_arr.min(axis=0), u_arr.max(axis=0)
    ranges = [
        np.arange(int((m - 1) * distinguishing_coefficient),
                  int((M + 1) * distinguishing_coefficient) + 2)
        for m, M in zip(min_u, max_u)
    ]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1)  # 形状 (nx, ny, nz, 3)
    grid = grid.reshape(-1, 3) / distinguishing_coefficient  # 转为 (N, 3)

    # 2. 在网格上找最大值
    vals = np.array([-f_3d(xyz, A, u_list, sigma) for xyz in grid])
    best_idx = np.argmin(vals)
    init_xyz = grid[best_idx]

    # 3. 精细优化
    res = minimize(lambda xyz: -f_3d(xyz, A, u_list, sigma),
                   x0=init_xyz, method='Powell', tol=1e-6)
    return res.x


def guided_filter(points, radius, epsilon):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
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

        points[i] = A @ points[i] + b

    return points


def remove_boundary_noise_normals(points, radius=0.1, angle_thresh=20):
    """
    基于法线变化剔除尖锐边界点
    points: (N,3)
    radius: 邻域搜索半径
    angle_thresh: 法线夹角阈值（度）
    """
    from sklearn.decomposition import PCA

    tree = cKDTree(points)
    mask = np.ones(len(points), dtype=bool)

    for i, pt in enumerate(points):
        idx = tree.query_ball_point(pt, radius)
        if len(idx) < 5:  # 跳过稀疏区域
            continue
        local_pts = points[idx]

        # PCA计算法线（最小特征向量）
        pca = PCA(n_components=3)
        pca.fit(local_pts - np.mean(local_pts, axis=0))
        normal = pca.components_[-1]

        # 与全局法线（如z轴）的夹角
        global_normal = np.array([0, 0, 1])
        angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(normal, global_normal)), 0, 1)))
        if angle > angle_thresh:
            mask[i] = False

    return points[mask]


def split_by_xy_ranges(points, ranges):
    """
    points : (N,3) ndarray
    ranges : list/tuple of [x_min, x_max, y_min, y_max]
    return points_in, points_out
    """
    ranges = np.asarray(ranges, dtype=float)  # (K,4)
    x_min, x_max, y_min, y_max = ranges.T  # 每个都是 (K,)

    # 广播比较，得到布尔矩阵 mask: (N,K)
    mask = (points[:, 0:1] >= x_min) & (points[:, 0:1] <= x_max) & \
           (points[:, 1:2] >= y_min) & (points[:, 1:2] <= y_max)

    in_any = mask.any(axis=1)  # (N,)
    points_in = points[in_any]
    points_out = points[~in_any]
    return points_in, points_out


def dbscan(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=10.0,  # 邻域距离
                                         min_points=100,  # 最小点数
                                         print_progress=True))  # 是否在控制台中可视化进度条
    print(labels.shape)
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    # ---------------------保存聚类结果------------------------
    communities = []
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    for i in range(max_label + 1):
        ind = np.where(labels == i)[0]
        clusters_cloud = pcd.select_by_index(ind)
        communities.append(clusters_cloud)
    print('cluster_nums:', len(communities))

    return communities


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
