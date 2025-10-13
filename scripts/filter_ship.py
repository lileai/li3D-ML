import open3d as o3d
import numpy as np
import os
import open3d.visualization.gui as gui
import glob
import time
from collections import Counter
from math import log


def statistical_outlier(pcd, nb_neighbors, std_ratio):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    clean_pcd = pcd.select_by_index(ind)
    return clean_pcd


def calculate_normals(pcd, radius=0.3, max_nn=100):
    # 计算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normal = np.asarray(pcd.normals)
    return normal


def calculate_pla(pcd, radius=0.3, max_nn=100):
    normal = calculate_normals(pcd, radius=radius, max_nn=max_nn)
    nz = normal[:, 2]
    pla_index = np.where(abs(nz) > 0.95)[0]
    pla_cloud = pcd.select_by_index(pla_index)
    return pla_cloud

def calculate_fac(pcd, radius=0.3, max_nn=100):
    normal = calculate_normals(pcd, radius=radius, max_nn=max_nn)
    nz = normal[:, 2]
    fac_index = np.where((nz > -0.05) & (nz < 0.05))[0]
    fac_cloud = pcd.select_by_index(fac_index)
    return fac_cloud


def calculate_splash(pcd):
    # 计算平均的高程(水花只会出现在平均高程以下)
    points = np.asarray((pcd.points))
    z_mean = points[:, 2].mean()
    process_index = np.where((points[:, 2] < z_mean))[0]
    no_process_index = np.where((points[:, 2] >= z_mean))[0]
    process_cloud = pcd.select_by_index(process_index)
    no_process_cloud = pcd.select_by_index(no_process_index)
    high_pcd = statistical_outlier(process_cloud, nb_neighbors=100, std_ratio=3.5)
    no_process_cloud = no_process_cloud + high_pcd
    return process_cloud, no_process_cloud


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

def get_mean_down_points(pcd):
    point = np.asarray((pcd.points))
    z_mean = point[:, 2].mean()
    process_index = np.where((point[:, 2] < z_mean))[0]
    process_cloud = pcd.select_by_index(process_index)
    return process_cloud

def get_mean_up_points(pcd):
    point = np.asarray((pcd.points))
    z_mean = point[:, 2].mean()
    process_index = np.where((point[:, 2] >= z_mean))[0]
    process_cloud = pcd.select_by_index(process_index)
    return process_cloud

def select_center(pcd):
    point = np.asarray((pcd.points))
    x_mean = point[:, 0].mean()
    process_index = np.where((point[:, 0] >= (x_mean - 20)) & (point[:, 0] <= (x_mean + 20)))[0]
    process_cloud = pcd.select_by_index(process_index)
    return process_cloud

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

def segment_plane(points, distance_threshold=0.01, ransac_n=3, num_iterations=100, probability=0.99999999):
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

def ransac(pcd, min_num=3300, dist=0.1, iters=0):
    segment = []  # 存储分割结果的容器
    points = np.asarray((pcd.points))
    while len(pcd.points) >= min_num:
        plane_model, inliers = segment_plane(points, distance_threshold=dist,
                                                 ransac_n=30,
                                                 num_iterations=2000)
        # plane_model, inliers = pcd.segment_plane(distance_threshold=dist,
        #                                          ransac_n=30,
        #                                          num_iterations=2000)
        plane_cloud = pcd.select_by_index(inliers)  # 分割出的平面点云
        r_color = np.random.uniform(0, 1, (1, 3))  # 平面点云随机赋色
        plane_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
        pcd = pcd.select_by_index(inliers, invert=True)  # 剩余的点云
        segment.append(plane_cloud)
        iters += 1
        if len(inliers) < min_num:
            break
    return segment

def process_single_ship(input_path, output_path, limit_max=70, calculate_dimensions=True):
    # 读取点云
    # with open(input_path, 'rb') as f:
    #     content = f.read()
    # print(content[:100])  # 打印文件的前100个字节
    pcd = o3d.io.read_point_cloud(input_path, format="pcd")
    points = np.asarray((pcd.points))
    print(points.shape[0])
    # # 去掉除船以外的其他点云
    index = np.where((points[:, 1] <= limit_max))[0]
    ship_cloud = pcd.select_by_index(index)
    ship_cloud = pcd
    fac_pcd = calculate_fac(ship_cloud)
    process_cloud_out_of_fac = calculate_completment(ship_cloud, fac_pcd)


    clean_pcd = ship_cloud
    # 取z轴四分位点
    process_cloud = get_mean_down_points(process_cloud_out_of_fac)
    process_cloud_down = get_mean_down_points(process_cloud)
    process_cloud = get_mean_up_points(process_cloud_down)
    # 取船中
    process_cloud = select_center(process_cloud)
    process_cloud_down_center = get_mean_down_points(process_cloud)
    pla_pcd = calculate_pla(process_cloud_down_center)
    segment = ransac(pcd=pla_pcd, min_num=len(pla_pcd.points))
    # 假设pla_clean_pcd是经过处理后的点云数据
    pla_clean_point = np.asarray(segment[0].points)
    # 提取所有的z坐标
    z_coords = np.round(pla_clean_point[:, 2], 3)
    # 统计每个z坐标出现的次数
    z_counts = Counter(z_coords)
    # 找出出现次数最多的z坐标
    most_common_z = z_counts.most_common(1)[0]  # 这会返回一个元组 (z_value, count)
    print(f"出现次数最多的z坐标是 {most_common_z[0]}，出现了 {most_common_z[1]} 次")
    pla_index = np.where(z_coords == most_common_z[0])[0]
    deck_pcd = segment[0]
    print(f"甲板相对于雷达的平均高度：{np.asarray(deck_pcd.points)[:, 2].mean()}")
    pcd_complement = calculate_completment(clean_pcd, deck_pcd)
    # 拟合平面并生成网格平面
    plane_model, inliers = deck_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model  # 平面方程 ax + by + cz + d =
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)

    # 获取点云的边界范围
    bounds = deck_pcd.get_axis_aligned_bounding_box()
    xmin, ymin, zmin = bounds.min_bound
    xmax, ymax, zmax = bounds.max_bound

    # 生成网格平面的顶点
    mesh_plane = o3d.geometry.TriangleMesh()
    mesh_plane.vertices = o3d.utility.Vector3dVector([
        [xmin, ymin, -(a * xmin + b * ymin + d) / c],
        [xmin, ymax, -(a * xmin + b * ymax + d) / c],
        [xmax, ymax, -(a * xmax + b * ymax + d) / c],
        [xmax, ymin, -(a * xmax + b * ymin + d) / c]
    ])

    # 生成网格平面的三角形面片
    mesh_plane.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],
        [0, 2, 3]
    ])

    # 给网格平面染色
    mesh_plane.paint_uniform_color([0, 1, 0])  # 绿色

    pcd_complement = calculate_completment(clean_pcd, deck_pcd)
    return (pcd,
            # ship_cloud,
            clean_pcd,
            pla_pcd,
            fac_pcd,
            segment,
            process_cloud_out_of_fac,
            pcd_complement,
            process_cloud_down,
            process_cloud_down_center,
            deck_pcd,
            mesh_plane)  # 返回网格平面
    # return pcd


if __name__ == '__main__':
    o3d.utility.random.seed(42)
    s = time.time()
    filename = 'ori_1750608952.4401906.pcd'
    # filename = 'ori_1750609757.9727232.pcd'
    # filename = 'ori_1750618920.0142221.pcd'
    # filename = 'ship_1749805571.460242_nan.pcd'
    dataset = '0618/pcd'
    directory_path = f"../data/{dataset}"
    out_path = f"{directory_path}/ship_out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 批量读取指定路径下的pcd文件名
    pcd_files = glob.glob(f'{directory_path}/*.pcd', recursive=True)
    # for filename in pcd_files:
    (pcd,
     # ship_cloud,
     clean_pcd,
     pla_pcd,
     fac_pcd,
     segment,
     process_cloud_out_of_fac,
     pcd_complement,
     process_cloud_down,
     process_cloud_down_center,
     deck_pcd,
     mesh_plane) = process_single_ship(
        input_path=f"{directory_path}/{filename}",
        output_path=f"{out_path}/clean_{filename}")
    print(len(deck_pcd.points))
    print(time.time() - s)
    deck_pcd.paint_uniform_color([1, 0, 0])
    pcd_complement.paint_uniform_color([1, 1, 1])
    process_cloud_out_of_fac.paint_uniform_color([0, 1, 0])
    np.asarray(pcd.colors)

    # ui
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("POINT", width=1600, height=1050)
    vis.show_skybox(False)
    vis.show_settings = True
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None

    vis.add_geometry("mesh_plane", mesh_plane)  # 添加网格平面
    vis.add_geometry("deck_pcd", deck_pcd)
    vis.add_geometry("pcd", pcd)
    for i, segment_item in enumerate(segment):
        vis.add_geometry(f"segment{i}", segment_item)
    vis.add_geometry("clean_pcd", clean_pcd)
    vis.add_geometry("pcd_complement", pcd_complement)
    vis.add_geometry("process_cloud_down", process_cloud_down)
    vis.add_geometry("process_cloud_down_center", process_cloud_down_center)
    vis.add_geometry("process_cloud_out_of_fac", process_cloud_out_of_fac)
    vis.add_geometry("pla_pcd", pla_pcd)
    vis.add_geometry("fac_pcd", fac_pcd)

    # set_viewer = True if os.path.exists('./view/customer_data_viewpoint.json_files') else False
    set_viewer = False
    if set_viewer:  # 自定义可视化初始视角参数
        param = o3d.io.read_pinhole_camera_parameters('./view/0618_viewpoint.json')
        extrinsic_matrix, intrinsic_matrix = param.extrinsic, param.intrinsic
        intrinsic_matrix, height, width = intrinsic_matrix.intrinsic_matrix, intrinsic_matrix.height, intrinsic_matrix.width
        vis.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)
    app.add_window(vis)
    app.run()
