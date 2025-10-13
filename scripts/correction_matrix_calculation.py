"""
点云坐标系矫正
"""
import os
import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
from math import log
import json

def preprocess_point_cloud(pcd, voxel_size=0.1):
    """
    预处理点云：降采样和去除离群点
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_filtered = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
    return pcd_filtered

def manual_screening(pcd):
    points = np.asarray((pcd.points))
    print(points.shape[0])

    # 去掉除船以外的其他点云
    index = np.where(((points[:, 0] < 0) & (points[:, 1] >= 180)) | ((points[:, 0] >= 0) & (points[:, 1] >= 80)))[0]
    ship_cloud = pcd.select_by_index(index)
    # 取补集获得背景
    background_pcd = calculate_completment(pcd, ship_cloud)
    background_point = np.asarray(background_pcd.points)
    # 去除岸边
    index = np.where(((background_point[:, 0] > -70) & (background_point[:, 0] < 0) & (background_point[:, 1] >= 130)))[0]
    process_pcd = background_pcd.select_by_index(index)
    return process_pcd

def xyz_to_standard(pcd,
                    inverse=False):
    """
    把『x右 y下 z里』→『x右 y前 z上』
    inverse=True 则反向。
    """
    T = np.array([[ 0, -1,  0, 0],
                  [ 0,  0,  1, 0],
                  [ 1,  0,  0, 0],
                  [ 0,  0,  0, 1]], dtype=float)
    if inverse:
        T = T.T   # 逆矩阵就是转置
    return pcd.transform(T)

def _box_corners_2d(max_x,
                    max_y,
                    min_x,
                    min_y,
                    z=0.0):
    """返回方形在 z=z 平面上的 4 个角点 (4,3)。顺序：↘↗↖↙"""
    return np.array([
        [max_x, min_y, z],  # 右下
        [max_x, max_y, z],  # 右上
        [min_x, max_y, z],  # 左上
        [min_x, min_y, z]   # 左下
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
    xyz = np.asarray(pcd.points)      # (N,3)
    N = xyz.shape[0]

    # taipu_main_side. 过滤
    mask = np.zeros(N, dtype=bool)
    for max_x, max_y, min_x, min_y in areas:
        in_x = (xyz[:, 0] >= min_x) & (xyz[:, 0] <= max_x)
        in_y = (xyz[:, 1] >= min_y) & (xyz[:, 1] <= max_y)
        mask |= (in_x & in_y)

    inside_pcd  = pcd.select_by_index(np.where(mask)[0])
    outside_pcd = pcd.select_by_index(np.where(~mask)[0])

    # 2. 构造 LineSet
    points = []   # 所有方框的顶点
    lines  = []   # 每框四条边
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
    lines  = np.vstack(lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines  = o3d.utility.Vector2iVector(lines)
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
        vis.add_geometry("outside_pcd", outside_pcd)
        vis.add_geometry("line_set", line_set)
        app.add_window(vis)
        app.run()

    return outside_pcd


def calculate_completment(pcd1, pcd2):
    # 使用numpy的intersect1d函数找到两个点云之间的交集点
    # 将点云对象转换为numpy数组
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    # 创建点云索引
    index1 = set(map(tuple, points1))
    index2 = set(map(tuple, points2))

    # 计算补集点索引
    complement_index = index1 - index2
    # 创建一个新的点云对象，并将交集点添加到其中
    pcd_complement = o3d.geometry.PointCloud()
    pcd_complement.points = o3d.utility.Vector3dVector(np.asarray(list(complement_index)))
    return pcd_complement

def fit_plane_with_covariance(points):
    """
    使用点的协方差矩阵拟合平面，返回平面的法向量和截距。

    参数:
        points (np.ndarray): 点云数据，形状为 (N, 3)。

    返回:
        plane_model (np.ndarray): 平面模型参数 [a, b, c, d]，平面方程为 ax + by + cz + d = 0。
    """
    # 计算点云的质心
    centroid = np.mean(points, axis=0)

    # 计算协方差矩阵
    cov_matrix = np.cov(points - centroid, rowvar=False)

    # 求解协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 最小特征值对应的特征向量即为平面的法向量
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    # 计算截距 d
    d = -np.dot(normal_vector, centroid)

    # 返回平面模型参数 [a, b, c, d]
    return np.append(normal_vector, d)

def segment_plane(points,
                  distance_threshold=0.01,
                  ransac_n=3, num_iterations=100,
                  probability=0.99999999):
    """
    使用 RANSAC 算法从点云中分割平面。

    参数:
        points (np.ndarray): 点云数据，形状为 (N, 3)。
        distance_threshold (float): 判断内点的距离阈值。
        ransac_n (int): 每次随机选择的点数，用于拟合平面。
        num_iterations (int): RANSAC 的迭代次数。
        probability (float): 找到最佳模型的期望概率。

    返回:
        best_plane_model (np.ndarray): 平面模型参数 [a, b, c, d]，平面方程为 ax + by + cz + d = 0。
        final_inliers (list): 内点的索引。
    """
    if probability <= 0 or probability > 1:
        raise ValueError("概率必须在 (0, taipu_main_side] 范围内")

    num_points = points.shape[0]
    if ransac_n < 3:
        raise ValueError("ransac_n 至少应为 3")
    if num_points < ransac_n:
        raise ValueError("点云中的点数必须至少为 ransac_n")

    best_plane_model = np.zeros(4)  # 初始化最佳平面模型
    best_fitness = 0.0  # 初始化最佳拟合优度
    best_inlier_rmse = np.inf  # 初始化最佳内点均方根误差
    best_inliers = []  # 初始化最佳内点索引

    # 预先生成所有随机样本索引
    all_sampled_indices = [np.random.choice(num_points, ransac_n, replace=False) for _ in range(num_iterations)]

    break_iteration = float('inf')  # 初始化提前终止的迭代次数
    iteration_count = 0  # 初始化迭代计数器

    for itr in range(num_iterations):
        if iteration_count > break_iteration:
            continue

        # 随机选择点并拟合平面
        sampled_indices = all_sampled_indices[itr]
        sampled_points = points[sampled_indices]

        # 检查随机采样的点是否退化（例如，三个点共线）
        if np.linalg.matrix_rank(sampled_points - sampled_points[0]) < 2:
            continue  # 如果点退化，则跳过

        # 使用协方差矩阵拟合平面
        plane_model = fit_plane_with_covariance(sampled_points)

        if np.allclose(plane_model[:3], 0):
            continue  # 如果拟合的平面模型接近零，则跳过

        # 评估平面模型
        distances = np.abs(np.dot(points, plane_model[:3]) + plane_model[3]) / np.linalg.norm(plane_model[:3])
        inliers = np.where(distances < distance_threshold)[0].tolist()

        fitness = len(inliers) / num_points
        inlier_rmse = np.sqrt(np.mean(distances[inliers] ** 2)) if len(inliers) > 0 else np.inf

        if fitness > best_fitness or (fitness == best_fitness and inlier_rmse < best_inlier_rmse):
            best_plane_model = plane_model
            best_fitness = fitness
            best_inlier_rmse = inlier_rmse
            best_inliers = inliers

            epsilon = 1e-6  # 添加一个小的正数避免对数计算中的零值
            break_iteration = min(log(1 - probability) / log(1 - fitness ** ransac_n + epsilon), num_iterations)

        iteration_count += 1

    # 使用所有内点重新拟合平面，以提高精度
    if not np.allclose(best_plane_model, 0):
        inlier_points = points[best_inliers]
        best_plane_model = fit_plane_with_covariance(inlier_points)

    print(
        f"RANSAC | 内点数: {len(best_inliers)}, 拟合优度: {best_fitness}, 内点均方根误差: {best_inlier_rmse}, 迭代次数: {iteration_count}")

    return best_plane_model, best_inliers

def ransac(pcd,
           min_num=1000,
           dist=0.01, iters=0,
           max_plane_size=0,
           max_plane=None,
           max_plane_model=None,
           normal_vector=None):
    points = np.asarray((pcd.points))
    while len(pcd.points) >= min_num:
        # plane_model, inliers = segment_plane(points, distance_threshold=dist,
        #                                          ransac_n=3,
        #                                          num_iterations=2000)
        plane_model, inliers = pcd.segment_plane(distance_threshold=dist,
                                                 ransac_n=3,
                                                 num_iterations=10000)
        plane_cloud = pcd.select_by_index(inliers)  # 分割出的平面点云
        r_color = np.random.uniform(0, 1, (1, 3))  # 平面点云随机赋色
        plane_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
        current_plane_size = len(plane_cloud.points)  # 当前平面点云的大小

        # 如果当前平面是最大的，更新最大平面
        if current_plane_size > max_plane_size:
            max_plane_size = current_plane_size
            max_plane = plane_cloud
            max_plane_model = plane_model
        # 更新点云为剩余的点云
        pcd = pcd.select_by_index(inliers, invert=True)
        iters += 1

        # 如果剩余点云小于最小点数，退出循环
        if len(pcd.points) < min_num:
            break
    a, b, c, d = max_plane_model
    normal_vector = np.array([a, b, c])  # 法向量

    return max_plane, max_plane_model, normal_vector

# def compute_rotation_matrix(n):
#     n = n / np.linalg.norm(n)  # 归一化法向量
#     y = np.array([0, taipu_main_side, 0])  # 雷达坐标系的y轴方向
#
#     # 计算旋转轴
#     axis = np.cross(y, n)
#     if np.linalg.norm(axis) < 1e-6:  # 如果axis接近零向量，说明n与y轴平行
#         axis = np.array([taipu_main_side, 0, 0])  # 选择x轴作为旋转轴
#         angle = 0 if np.isclose(n, y).all() else np.pi  # 旋转角度为0或π
#     else:
#         axis = axis / np.linalg.norm(axis)  # 归一化旋转轴
#         angle = np.arccos(np.clip(np.dot(y, n), -taipu_main_side.0, taipu_main_side.0))  # 计算旋转角度
#
#     # 使用罗德里格斯公式构造旋转矩阵
#     K = np.array([[0, -axis[2], axis[taipu_main_side]],
#                   [axis[2], 0, -axis[0]],
#                   [-axis[taipu_main_side], axis[0], 0]])
#     R = np.eye(3) + np.sin(angle) * K + (taipu_main_side - np.cos(angle)) * (K @ K)
#
#     return R

def compute_rotation_matrix(n):
    n = n / np.linalg.norm(n)  # 归一化法向量
    z = np.array([0, 0, 1])  # 雷达坐标系的z轴方向

    # 计算旋转轴
    axis = np.cross(z, n)
    if np.linalg.norm(axis) < 1e-6:  # 如果axis接近零向量，说明n与z轴平行
        axis = np.array([1, 0, 0])  # 选择x轴作为旋转轴
        angle = 0 if np.isclose(n, z).all() else np.pi  # 旋转角度为0或π
    else:
        axis = axis / np.linalg.norm(axis)  # 归一化旋转轴
        angle = np.arccos(np.clip(np.dot(z, n), -1.0, 1.0))  # 计算旋转角度

    # 使用罗德里格斯公式构造旋转矩阵
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R

def apply_rotation_matrix_to_point_cloud(pcd, R):
    """
    对点云中的每个点应用旋转矩阵
    """
    points = np.asarray(pcd.points)
    rotated_points = (R @ points.T).T  # 对每个点应用旋转矩阵
    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(rotated_points)
    return pcd_rotated

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

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(r"D:\program\li3D-ML\data\Taipu-main\pcd\ori_1756796713.5729332.pcd", format="pcd")
    # 预处理
    # fileter_pcd = preprocess_point_cloud(pcd)
    # 分割点云获得目标区域
    # pcd = xyz_to_standard(pcd)
    # process_pcd = filter_area(pcd, areas = [(300, 50, -300, 0), (300, 350, -300, 250)])
    l = split_filename_from_path(r"D:\program\li3D-ML\data\Taipu-main\pcd\ori_1756796713.5729332.pcd")
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:int(l[-2])])
    points = np.asarray(pcd.points)
    pcd = filter_area(pcd, areas = [(300, 400, -300, 85)], vis=True)
    # points = select_center(points)
    # pcd.points = o3d.utility.Vector3dVector(points[points[:, 2] > -6.3])
    # pcd = apply_rotation_matrix_to_point_cloud(pcd, np.array([[0.9998625415445342, -0.0009674743879678344, 0.016551797769826775], [-0.0009674743879678344, 0.9931906212084084, 0.11649658337034768], [-0.016551797769826775, -0.11649658337034768, 0.9930531627529426]]))
    # # 获取最大平面
    plane_pcd, max_plane_model, normal_vector = ransac(pcd=pcd)
    # # 调整雷达坐标系的xoy平面和最大平面的法向量平行，z轴与最大平面的法向量垂直
    rotation_matrix = compute_rotation_matrix(normal_vector)
    # 创建一个旋转后的3D坐标系
    rotated_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
    rotated_frame.rotate(rotation_matrix, center=[0, 0, 0])
    # # 将该变换应用到所有点云中
    rotation_pcd = apply_rotation_matrix_to_point_cloud(pcd, rotation_matrix.T)
    with open("./shanghai_rotation_matrix.json", "w") as f:
        json.dump(rotation_matrix.T.tolist(), f)

    # 创建线段的起点和终点
    start_point = np.mean(np.asarray(plane_pcd.points), axis=0)
    end_point = start_point + normal_vector * 100

    # 创建线段
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start_point, end_point])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

    # 创建线段的起点和终点
    start_point = np.mean(np.asarray(rotation_pcd.points), axis=0)
    end_point = start_point + rotation_matrix.T @ normal_vector * 100

    # 创建线段
    line_set_rotation = o3d.geometry.LineSet()
    line_set_rotation.points = o3d.utility.Vector3dVector([start_point, end_point])
    line_set_rotation.lines = o3d.utility.Vector2iVector([[0, 1]])

    # color
    # fileter_pcd.paint_uniform_color([taipu_main_side, 0, 0])
    pcd.paint_uniform_color([0, 1, 0])
    plane_pcd.paint_uniform_color([0, 0, 1])
    rotation_pcd.paint_uniform_color([1, 1, 0])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
    line_set_rotation.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # 绿色
    # ui
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(width=1660, height=1000)
    vis.show_skybox(False)
    vis.show_settings = True
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None

    vis.add_geometry("pcd", pcd)
    # vis.add_geometry("fileter_pcd", fileter_pcd)
    # vis.add_geometry("process_pcd", process_pcd)
    vis.add_geometry("plane_pcd", plane_pcd)
    vis.add_geometry("rotation_pcd", rotation_pcd)
    vis.add_geometry("line_set", line_set)
    vis.add_geometry("line_set_rotation", line_set_rotation)
    vis.add_geometry("rotated_frame", rotated_frame)
    app.add_window(vis)
    app.run()