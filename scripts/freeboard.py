"""
干舷法提交版本
"""
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import entropy
from sklearn.decomposition import PCA


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


def statistical_outlier(points, nb_neighbors=16, std_ratio=1.0, is_inliers=False):
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
    return principal


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


def compute_average_density(points, k_neighbors=8):
    if points.size == 0:
        return 0.0
    kdtree = cKDTree(points)
    distances, indices = kdtree.query(points, k=k_neighbors)
    avg_density = k_neighbors / np.mean(distances[:, -1])
    return avg_density


def compute_roughness_entropy(points, k_neighbors=8, radius=1.0):
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

    principal_direction = compute_principal_direction(selected_part)

    # 3. 构建旋转矩阵并应用
    rotation_matrix = build_rotation_matrix(principal_direction)
    center = np.mean(center_points, axis=0)
    rotated_points = np.dot(center_points - center, rotation_matrix.T) + center
    rotation_total_points = np.dot(points - center, rotation_matrix.T) + center

    return rotation_total_points, rotated_points, rotation_matrix, center


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

def freeboard_calculation(points, water_level=None, parser_dict=None):
    '''

    :param points:点云
    :param water_level: 水位
    :param parser_dict:
{
    "func_mode": "[bridge, bank]桥基模式和岸基模式",
    "x_min": "船中心的最小计算范围，桥基模式下是一个正值，岸基模式下是一个负值",
    "x_max": "船中心的最大计算范围",
    "voxel_size": "体素化的网格尺寸",
    "points_thresh": "半船点的最小点数",
    "density_thresh": "半船点的最小体密度",
    "save_ratio": "边缘点保存的比例，越小边缘点的范围越小",
    "filter_thresh": "船中点云需要滤波的最小点数",
    "nb_neighbors": "滤波邻域点个数",
    "std_ratio": "滤波标准差",
    "divide_thresh": "分段计算的最小单元",
    "axis_offset": "坐标轴偏移距离",
    "var_thresh": "船舶边缘y值的方差"
  }
    :return:
    '''
    o3d.utility.random.seed(42)  # 设置随机种子为固定值
    points = voxel_filter(points, voxel_size=parser_dict["voxel_size"])
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    x_center = (x_max + x_min) / 2
    if x_center < parser_dict['x_min'] or x_center > parser_dict['x_max']:
        return np.nan, np.nan, np.nan
    center_points = select_center(points)
    density = compute_average_density(center_points, k_neighbors=16)
    (
        rotation_total_points,
        rotation_center_points,
        rotation_matrix,
        center,  # 旋转中心
    ) = apply_rotation(points, center_points, density,
                           points_thresh=parser_dict['points_thresh'],
                           density_thresh=parser_dict['density_thresh'],
                           mode=parser_dict['func_mode'])
    # 将rotation_center_points按x值分成n等分
    boundary_points = detect_boundaries(rotation_center_points, radius=0.5 * density, k=16,
                                        angle_threshold=np.pi * 0.7)
    process_points, direction = filter_by_y(boundary_points, rotation_total_points,
                                            mode=parser_dict['func_mode'],
                                            save_ratio=parser_dict['save_ratio'])
    if len(process_points) > parser_dict['filter_thresh']:
        nb_neighbors = parser_dict['nb_neighbors'] * density
        std_ratio = parser_dict['std_ratio'] * density
        process_points, _ = mahalanobis_filter_vec(process_points, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    n = max(1, int(len(process_points) / parser_dict['divide_thresh']))
    divide_rotation_center_list = divide_points(process_points, axis=0, n=n)
    freeboard_upper_points = []
    for divide_rotation_center in divide_rotation_center_list:
        # 对每个部分提取y坐标
        y_coords = divide_rotation_center[:, 1] + parser_dict['axis_offset']
        # taipu_main_side. 计算排序索引（升序）
        score = direction * np.abs(y_coords)
        order = np.argsort(score)
        # 2. 逐步扩大窗口，直到方差超标
        for k in range(1, len(order) + 1):
            idx_win = order[:k]
            var = np.var(y_coords[idx_win])
            if var >= parser_dict['var_thresh']:
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
    # 拿出freeboard_upper_points保留两位小数后频数最多的点作为freeboard_upper_point
    freeboard_upper_points = np.asarray(freeboard_upper_points)
    freeboard_upper_points = reverse_rotation(freeboard_upper_points, rotation_matrix, center)
    freeboard_upper_bound = find_most_frequent_value(freeboard_upper_points[:, 2], precision=2, n=n, threshold=0.1)
    if freeboard_upper_bound is None:
        freeboard_upper_bound = np.median(freeboard_upper_points[:, 2])
    if water_level is not None:
        freeboard_lower_bound = water_level
        deck = abs(freeboard_upper_bound - freeboard_lower_bound)
    else:
        freeboard_lower_bound = center_points[:, 2].min()
        deck = freeboard_upper_bound - freeboard_lower_bound
    return (
        freeboard_upper_bound,
        freeboard_lower_bound,
        deck,
    )
