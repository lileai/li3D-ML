#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只对动态点建立 Pose Graph（粗/精 + 全局优化）
可视化：第 0 帧完整，其余帧仅动态点
python dynamic_pose_graph_view.py ../data/flow_leishen --thresh taipu_main_side.0
"""
import copy
import os
import time
import glob
import argparse
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial import cKDTree

# 调整参数以提高鲁棒性
FLOW_THRESH = 0.15  # 稍微增加动态点阈值
VOXEL_SIZE = 0.2    # 减小体素大小以提高精度
MAX_CORR_COARSE = 15 * VOXEL_SIZE
MAX_CORR_FINE = VOXEL_SIZE
MIN_OVERLAP_RATIO = 0.08  # 稍微提高最小重叠度要求
MIN_POINTS_FOR_REGISTRATION = 20  # 增加最小点数要求


def build_box(points, deck_z=3.5):
    """
    构建并绘制三维点云的包围盒。

    参数:
    points: 三维点云，形状为 (n, 3) 的 NumPy 数组。
    """
    # 计算包围盒的最小和最大坐标
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)
    min_x -= 1
    min_z -= 1
    max_x += 1
    max_z += 1
    print(max_z)

    duan = 0.3
    vertices = np.array([
        [min_x - duan, min_y, min_z + deck_z],  # a0
        [min_x, min_y, min_z + deck_z],  # b1
        [min_x + duan, min_y, min_z + deck_z],  # c2
        [min_x - duan, min_y, min_z + 1],  # d3
        [min_x, min_y, min_z + 1],  # e4
        [min_x + duan, min_y, min_z + 1],  # f5
        [min_x + 1, min_y, min_z + duan],  # g6
        [min_x + 1, min_y, min_z],  # h7
        [min_x + 1, min_y, min_z - duan],  # i8
        [max_x - 1, min_y, min_z + duan],  # j9
        [max_x - 1, min_y, min_z],  # k10
        [max_x - 1, min_y, min_z - duan],  # l11
        [max_x + duan, min_y, min_z + 1],  # m12
        [max_x, min_y, min_z + 1],  # n13
        [max_x - duan, min_y, min_z + 1],  # o14
        [max_x - duan, min_y, max_z],  # p15
        [max_x, min_y, max_z],  # q16
        [max_x + duan, min_y, max_z],  # r17
    ])

    edges = [
        [0, 1], [1, 2], [1, 4], [3, 4], [4, 5],  # 高
        [6, 7], [7, 8], [7, 10], [9, 10], [10, 11],  # 长
        [12, 13], [13, 14], [13, 16], [15, 16], [16, 17]

    ]

    # 创建线对象
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(vertices)
    box.lines = o3d.utility.Vector2iVector(edges)  # 使用 edges 的索引对
    box.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(edges))])  # 设置为红色

    return box


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


def estimate_normals_vec(points: np.ndarray,
                         radius: float,
                         max_nn: int) -> np.ndarray:
    N = points.shape[0]
    tree = cKDTree(points)

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

    # === 关键：加正则化防止奇异 ===
    eps = 1e-6
    cov += np.eye(3) * eps

    # 稳健 SVD，捕获不收敛
    normals = np.zeros_like(points)
    try:
        _, _, Vt = np.linalg.svd(cov)
        normals[mask] = Vt[mask, -1, :]
    except np.linalg.LinAlgError:
        # 退化点直接给零向量，后续调用端可再平滑处理
        pass

    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals


def load_pcd(path):
    l = split_filename_from_path(path)
    pcd = o3d.io.read_point_cloud(path)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:int(l[-2])])
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    return pcd


def get_dynamic_points(src, dst, thresh):
    pts_src = np.asarray(src.points)
    pts_dst = np.asarray(dst.points)
    tree = KDTree(pts_dst)
    dist, idx = tree.query(pts_src, k=1)
    delta = pts_src - pts_dst[idx]
    mask = np.linalg.norm(delta, axis=1) > thresh
    return mask

# SVD分解计算点云的质心与特征向量
def pca_compute(data):
    [center, covariance] = data.compute_mean_and_covariance()
    # SVD奇异值分解，得到covariance矩阵的特征值和特征向量
    eigenvectors, _, _ = np.linalg.svd(covariance)
    return eigenvectors, center


# PCA实现点云配准
def pca_registration(P, X):

    P.paint_uniform_color([1, 0, 0])  # 给原始点云赋色
    X.paint_uniform_color([0, 1, 0])

    error = []   # 定义误差集合
    matrax = []  # 定义变换矩阵集合
    Up, Cp = pca_compute(P)  # PCA方法得到P对应的特征向量、点云质心
    Ux, Cx = pca_compute(X)  # PCA方法得到X对应的特征向量、点云质心
    # 主轴对应可能出现的情况
    Upcopy = Up
    sign1 = [1, -1, 1, 1, -1, -1, 1, -1]
    sign2 = [1, 1, -1, 1, -1, 1, -1, -1]
    sign3 = [1, 1, 1, -1, 1, -1, -1, -1]
    for nn in range(len(sign3)):
        Up[0] = sign1[nn]*Upcopy[0]
        Up[1] = sign2[nn]*Upcopy[1]
        Up[2] = sign3[nn]*Upcopy[2]
        R0 = np.dot(Ux, np.linalg.inv(Up))
        T0 = Cx-np.dot(R0, Cp)
        T = np.eye(4)
        T[:3, :3] = R0
        T[:3, 3] = T0
        T[3, 3] = 1
# 计算配准误差，误差最小时对应的变换矩阵即为最终完成配准的变换
        trans = copy.deepcopy(P).transform(T)
        dists = trans.compute_point_cloud_distance(X)
        dists = np.asarray(dists)  # 欧氏距离（单位是：米）
        mse = np.average(dists)
        error.append(mse)
        matrax.append(T)
    min_error = min(error)
    ind = error.index(min_error)  # 获取误差最小时对应的索引
    final_T = matrax[ind]  # 根据索引获取变换矩阵
    return final_T, min_error

def dynamic_pairwise_registration(src, dst, thresh, mode=None):
    """
    优化后的动态点云配准函数
    """
    # 1. 动态点提取 & 重叠度检查
    if mode == "flow":
        pts_src = np.asarray(src.points)
        pts_dst = np.asarray(dst.points)

        mask_src = get_dynamic_points(src, dst, thresh)
        mask_dst = get_dynamic_points(dst, src, thresh)

        # 更严格的重叠度检查
        overlap_ratio = min(mask_src.sum(), mask_dst.sum()) / max(len(pts_src), len(pts_dst), 1)
        if (overlap_ratio < MIN_OVERLAP_RATIO or
                mask_src.sum() < MIN_POINTS_FOR_REGISTRATION or
                mask_dst.sum() < MIN_POINTS_FOR_REGISTRATION):
            print(f"[WARN] 重叠度={overlap_ratio:.3f} 或点数不足，跳过该帧")
            T = np.eye(4)
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                src, dst, MAX_CORR_FINE, np.eye(4))
            return T, info, float('inf')

        src_dyn = src.select_by_index(np.where(mask_src)[0])
        dst_dyn = dst.select_by_index(np.where(mask_dst)[0])
    else:
        src_dyn = src
        dst_dyn = dst

    # 法线估计
    src_normals = estimate_normals_vec(np.asarray(src_dyn.points), radius=VOXEL_SIZE * 2, max_nn=30)
    src_dyn.normals = o3d.utility.Vector3dVector(src_normals)
    dst_normals = estimate_normals_vec(np.asarray(dst_dyn.points), radius=VOXEL_SIZE * 2, max_nn=30)
    dst_dyn.normals = o3d.utility.Vector3dVector(dst_normals)

    # ---------- 3. 自适应 FPFH 参数 ----------
    def calc_fpfh_parameters(pcd):
        points = np.asarray(pcd.points)
        if len(points) < 100:  # 极稀疏
            return VOXEL_SIZE * 5.0
        # 估算每体素点数
        volume = (points.max(0) - points.min(0)).prod()
        points_per_voxel = len(points) / (volume / (VOXEL_SIZE ** 3) + 1e-6)
        if points_per_voxel < 10:  # 稀疏
            return VOXEL_SIZE * 5.0
        elif points_per_voxel < 30:  # 中等
            return VOXEL_SIZE * 4.0
        else:  # 稠密
            return VOXEL_SIZE * 3.0

    # 3. 自适应 FPFH 参数
    fpfh_radius_src = calc_fpfh_parameters(src_dyn)
    fpfh_radius_dst = calc_fpfh_parameters(dst_dyn)

    # 使用更保守的参数选择策略
    fpfh_radius = max(fpfh_radius_src, fpfh_radius_dst)
    fpfh_nn = 100

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_dyn,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_nn))
    dst_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        dst_dyn,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_nn))

    # 3. 多策略粗配准 —— 策略0：FGR（无初值）
    best_result = None
    try:
        fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_dyn, dst_dyn, src_fpfh, dst_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=MAX_CORR_COARSE))
        if fgr_result.transformation.trace() != 0:          # 排除零矩阵
            best_result = fgr_result
    except Exception as e:
        print(f"FGR 配准失败: {e}")

    if best_result is None:                                 # ★ 保护
        print("粗配准完全失败")
        T = np.eye(4)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_dyn, dst_dyn, MAX_CORR_FINE, np.eye(4))
        return T, info, float('inf')

    # T_pca, pca_rmse = pca_registration(src_dyn, dst_dyn)
    T_coarse =  best_result.transformation

    # 精配准 - ICP
    current_max_corr_dist = MAX_CORR_FINE

    fine = o3d.pipelines.registration.registration_generalized_icp(source=src_dyn,
                                                       target=dst_dyn,
                                                       max_correspondence_distance=current_max_corr_dist,
                                                       init=T_coarse,
                                                       estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

    T_constrained = fine.transformation

    # 计算信息矩阵
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src_dyn, dst_dyn, current_max_corr_dist, T_constrained)

    return T_constrained, info, fine.inlier_rmse


def build_dynamic_pose_graph(pcds, thresh):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    # 1. 帧号→节点号 映射表
    frm2node = [-1] * len(pcds)
    frm2node[0] = 0

    keyframe_indices = [0]
    # 2. 世界→上一帧里程计（初始化）
    Twf_prev = np.eye(4)

    for i in range(1, len(pcds)):
        T_k2c, info, rmse = dynamic_pairwise_registration(
            pcds[i], pcds[keyframe_indices[-1]], thresh, mode="flow")

        print(f"Frame {keyframe_indices[-1]}→{i}  RMSE={rmse:.4f}")

        # 3. 世界→当前帧
        k = keyframe_indices[-1]
        T_w_k = np.linalg.inv(pose_graph.nodes[frm2node[k]].pose)
        Twf = T_w_k @ T_k2c

        # 4. 关键帧判定（用相对运动）
        delta = np.linalg.inv(Twf_prev) @ Twf
        trans = np.linalg.norm(delta[:3, 3])
        rot   = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))

        if trans > 1.0 or rot > 0.3:
            print("→ 关键帧")
            keyframe_indices.append(i)
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(Twf)))
            frm2node[i] = len(pose_graph.nodes) - 1

            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    frm2node[keyframe_indices[-2]], frm2node[i], T_k2c, info, uncertain=False))

        Twf_prev = Twf.copy()   # 5. 保存上一帧世界位姿

    return pose_graph, keyframe_indices, frm2node   # 6. 返回映射表


def run_icp_registration(pcd_dir=None, pcd_list=None, verbose=False):
    if pcd_dir is not None:
        files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
        assert len(files) >= 2
        start_time = time.time()
        pcds = [load_pcd(f) for f in files]
    elif pcd_list is not None:
        pcds = pcd_list
    # 构建 & 优化 Pose Graph（仅用动态点）
    pose_graph, keyframe_indices, _ = build_dynamic_pose_graph(pcds, FLOW_THRESH)

    print("Optimizing Pose Graph …")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=MAX_CORR_FINE,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    # ---------- 可视化 ----------
    # 第 0 帧完整
    pcd = o3d.geometry.PointCloud()
    base = pcds[0]
    base.paint_uniform_color([0.7, 0.7, 0.7])
    pcd += base
    print(time.time() - start_time)
    # 后续帧只取动态点
    colors = plt.cm.jet(np.linspace(0, 1, len(pcds)))[:, :3]
    transformed_pcds = [base]
    for index, i in enumerate(keyframe_indices):
        mask = get_dynamic_points(pcds[i], pcds[i - 1], FLOW_THRESH)
        dyn_pts = pcds[i].select_by_index(np.where(mask)[0])
        if len(dyn_pts.points) == 0:
            continue
        dyn_pts = pcds[i].select_by_index(np.where(mask)[0])
        T = np.linalg.inv(pose_graph.nodes[index].pose)
        transformed = dyn_pts.transform(T)
        transformed.paint_uniform_color(colors[i])
        transformed_pcds.append(transformed)
        pcd += transformed

    o3d.io.write_point_cloud(f'lidar_combine.pcd', pcd, print_progress=False)
    print(time.time() - start_time)
    if verbose:
        # 可视化网格
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Mesh", width=1024, height=768)
        vis.show_skybox(False)
        vis.show_settings = True
        # 设置背景颜色为黑色
        background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
        vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None
        # vis.add_geometry("mesh", mesh)
        for i, pcd in enumerate(transformed_pcds):
            vis.add_geometry(f"pcd{i}", pcd)
        vis.add_geometry("pcd", pcd)
        # vis.add_geometry("box", box)
        vis.setup_camera(
            45.0,  # fov
            np.array([0, 0, 0], dtype=np.float32),  # center
            np.array([10, 0, 0], dtype=np.float32),  # eye
            np.array([0, 0, 1], dtype=np.float32)  # up
        )
        app.add_window(vis)
        app.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_dir', default="../../data/flow_test",
                        help='目录 *.pcd')
    args = parser.parse_args()
    run_icp_registration(args.pcd_dir, verbose=True)



