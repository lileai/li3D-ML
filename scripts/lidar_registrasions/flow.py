#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只对动态点建立 Pose Graph（关键帧 + 全局优化）
可视化：第 0 帧完整，其余帧仅动态点
普通帧通过插值得到优化后位姿，也用于融合
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
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# 调整参数以提高鲁棒性
FLOW_THRESH = 0.15
VOXEL_SIZE = 0.4
MAX_CORR_COARSE = 10 * VOXEL_SIZE
MAX_CORR_FINE = 0.4
MIN_OVERLAP_RATIO = 0.08
MIN_POINTS_FOR_REGISTRATION = 20
FPFH_NN = 100
SIGMA = 0.5



def split_filename_from_path(file_path):
    file_name_with_extension = os.path.basename(file_path)
    file_name, _ = os.path.splitext(file_name_with_extension)
    parts = file_name.split('_')
    return [part for part in parts if part]


def estimate_normals_vec(points: np.ndarray, radius: float, max_nn: int) -> np.ndarray:
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
    eps = 1e-6
    cov += np.eye(3) * eps

    normals = np.zeros_like(points)
    try:
        _, _, Vt = np.linalg.svd(cov)
        normals[mask] = Vt[mask, -1, :]
    except np.linalg.LinAlgError:
        pass

    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals


def load_pcd(path):
    """
    加载点云，仅根据文件名裁剪点数，不进行离群点去除。
    返回：
        pcd_raw: 裁剪后的原始点云（未体素化，未去噪）
        pcd_voxel: 裁剪后并体素下采样（用于配准等）
    """
    l = split_filename_from_path(path)
    pcd = o3d.io.read_point_cloud(path)
    num_points = int(l[-2])
    points = np.asarray(pcd.points)
    if len(points) > num_points:
        points = points[:num_points]
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(points)

    pcd_voxel = pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    return pcd_raw, pcd_voxel


def get_dynamic_points(src, dst, thresh):
    pts_src = np.asarray(src.points)
    pts_dst = np.asarray(dst.points)
    tree = KDTree(pts_dst)
    dist, idx = tree.query(pts_src, k=1)
    delta = pts_src - pts_dst[idx]
    mask = np.linalg.norm(delta, axis=1) > thresh
    return mask


def dynamic_pairwise_registration(src, dst, thresh, mode=None):
    if mode == "flow":
        pts_src = np.asarray(src.points)
        pts_dst = np.asarray(dst.points)
        mask_src = get_dynamic_points(src, dst, thresh)
        mask_dst = get_dynamic_points(dst, src, thresh)
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

    src_normals = estimate_normals_vec(np.asarray(src_dyn.points), radius=VOXEL_SIZE * 2, max_nn=30)
    src_dyn.normals = o3d.utility.Vector3dVector(src_normals)
    dst_normals = estimate_normals_vec(np.asarray(dst_dyn.points), radius=VOXEL_SIZE * 2, max_nn=30)
    dst_dyn.normals = o3d.utility.Vector3dVector(dst_normals)

    def calc_fpfh_parameters(pcd):
        points = np.asarray(pcd.points)
        if len(points) < 1000:
            return VOXEL_SIZE * 5.0
        volume = (points.max(0) - points.min(0)).prod()
        points_per_voxel = len(points) / (volume / (VOXEL_SIZE ** 3) + 1e-6)
        if points_per_voxel < 10:
            return VOXEL_SIZE * 5.0
        elif points_per_voxel < 30:
            return VOXEL_SIZE * 4.0
        else:
            return VOXEL_SIZE * 3.0

    fpfh_radius = max(calc_fpfh_parameters(src_dyn), calc_fpfh_parameters(dst_dyn))

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_dyn, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=FPFH_NN))
    dst_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        dst_dyn, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=FPFH_NN))

    best_result = None
    try:
        fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_dyn, dst_dyn, src_fpfh, dst_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=MAX_CORR_COARSE))
        if fgr_result.transformation.trace() != 0:
            best_result = fgr_result
    except Exception as e:
        print(f"FGR 配准失败: {e}")

    if best_result is None:
        print("粗配准完全失败")
        T = np.eye(4)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_dyn, dst_dyn, MAX_CORR_FINE, np.eye(4))
        return T, info, float('inf')
    loss = o3d.pipelines.registration.HuberLoss(SIGMA)
    T_coarse = best_result.transformation
    fine = o3d.pipelines.registration.registration_generalized_icp(
        src_dyn, dst_dyn, MAX_CORR_FINE, T_coarse,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    transformation = fine.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src_dyn, dst_dyn, 0.1, transformation)
    return transformation, info, fine.inlier_rmse


def build_dynamic_pose_graph(pcds, thresh):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    keyframe_indices = [0]
    T_world_curr = np.eye(4)
    estimated_poses = [np.eye(4)]  # 每一帧的初始估计位姿（世界坐标系）

    for i in range(1, len(pcds)):
        ref_idx = keyframe_indices[-1]
        T_rel, info, rmse = dynamic_pairwise_registration(pcds[i], pcds[ref_idx], thresh, mode="flow")
        print(f"Frame {ref_idx}→{i} RMSE={rmse:.4f}")

        T_world_curr = T_rel @ T_world_curr
        estimated_poses.append(T_world_curr.copy())

        trans_norm = np.linalg.norm(T_rel[:3, 3])
        rot_rad = np.arccos(np.clip((np.trace(T_rel[:3, :3]) - 1) / 2, -1, 1))
        if trans_norm > 1.0 or rot_rad > 0.3:
            print(f"→ 关键帧 {i}")
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(T_world_curr)))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    len(pose_graph.nodes) - 2,
                    len(pose_graph.nodes) - 1,
                    T_rel, info, uncertain=False))
            keyframe_indices.append(i)

    return pose_graph, keyframe_indices, estimated_poses


def interpolate_poses(keyframe_node_poses, keyframe_indices, total_frames):
    """
    keyframe_node_poses: list of node.pose (which is inv(T_world))
    """
    interpolated = [None] * total_frames
    # Step 1: fill keyframes
    for i, idx in enumerate(keyframe_indices):
        T_opt = np.linalg.inv(keyframe_node_poses[i])  # node stores inv(T)
        interpolated[idx] = T_opt

    # Step 2: interpolate between keyframes
    for seg in range(len(keyframe_indices) - 1):
        i0 = keyframe_indices[seg]
        i1 = keyframe_indices[seg + 1]
        T0 = interpolated[i0]
        T1 = interpolated[i1]
        r0 = R.from_matrix(T0[:3, :3])
        r1 = R.from_matrix(T1[:3, :3])
        t0, t1 = T0[:3, 3], T1[:3, 3]

        for j in range(i0 + 1, i1):
            ratio = (j - i0) / (i1 - i0)
            t_interp = (1 - ratio) * t0 + ratio * t1
            r_interp = r0.slerp(r1, ratio)
            T_interp = np.eye(4)
            T_interp[:3, :3] = r_interp.as_matrix()
            T_interp[:3, 3] = t_interp
            interpolated[j] = T_interp

    # Step 3: fill trailing frames (after last keyframe)
    last_kf = keyframe_indices[-1]
    last_T = interpolated[last_kf]
    for j in range(last_kf + 1, total_frames):
        interpolated[j] = last_T

    # Safety: ensure no None
    for i in range(total_frames):
        if interpolated[i] is None:
            interpolated[i] = np.eye(4)
    return interpolated


# 6. 约束自由度 - 更平滑的约束
def enforce_x_yaw(T):
    """仅保留 x 平移 + yaw 旋转，但使用更平滑的约束"""
    # 提取当前位姿参数
    R = T[:3, :3]
    t = T[:3, 3]

    # 计算yaw角
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])

    def build_scaled_rotmat(yaw, pitch, roll,
                            yaw_scale=1.0,
                            pitch_scale=0.1,
                            roll_scale=0.0):
        """
        按给定缩放系数重新生成旋转矩阵（Z-Y-X 顺序）
        """
        y = yaw * yaw_scale
        p = pitch * pitch_scale
        r = roll * roll_scale

        cy, sy = np.cos(y), np.sin(y)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)

        Rz = np.array([[cy, -sy, 0],
                       [sy, cy, 0],
                       [0, 0, 1]])

        Ry = np.array([[cp, 0, sp],
                       [0, 1, 0],
                       [-sp, 0, cp]])

        Rx = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr, cr]])

        return Rz @ Ry @ Rx

    # 构建新旋转矩阵
    R_new = build_scaled_rotmat(yaw, pitch, roll,
                                yaw_scale=1.0,
                                pitch_scale=0.01,
                                roll_scale=0.01)

    # 保留x平移
    t_new = np.array([t[0],
                      t[1] * 0.1,  # 允许少量y移动
                      t[2] * 0.1])  # 允许少量z移动

    T_constrained = np.eye(4)
    T_constrained[:3, :3] = R_new
    T_constrained[:3, 3] = t_new

    return T_constrained


def run_icp_registration(pcd_dir=None, pcd_list=None, verbose=False):
    if pcd_dir is not None:
        files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
        assert len(files) >= 2, "Need at least 2 PCD files"
        start_time = time.time()
        loaded = [load_pcd(f) for f in files]
        pcds_raw = [item[0] for item in loaded]  # 原始裁剪点云（不去噪）
        pcds_voxel = [item[1] for item in loaded]  # 体素下采样点云
    elif pcd_list is not None:
        pcds_raw = pcd_list
        pcds_voxel = pcd_list
    else:
        raise ValueError("Either pcd_dir or pcd_list must be provided")

    # Step 1: Build pose graph using keyframes only
    pose_graph, keyframe_indices, _ = build_dynamic_pose_graph(pcds_voxel, FLOW_THRESH)

    # Step 2: Global optimization
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

    # Step 3: Interpolate all frame poses
    optimized_keyframe_poses = [node.pose for node in pose_graph.nodes]
    all_optimized_poses = interpolate_poses(
        optimized_keyframe_poses, keyframe_indices, len(pcds_raw))

    # Step 4: Fuse all frames (including non-keyframes)
    pcd = o3d.geometry.PointCloud()
    colors = plt.cm.jet(np.linspace(0, 1, len(pcds_raw)))[:, :3]
    transformed_pcds = []
    for i in range(len(pcds_raw)):
        if i == 0:
            base = copy.deepcopy(pcds_raw[0])
            T_opt = all_optimized_poses[0]
            T_opt = enforce_x_yaw(T_opt)  # 可选
            pcd += base.transform(T_opt)
        else:
            # mask = get_dynamic_points(pcds[i], pcds[i - 1], FLOW_THRESH)
            # # dyn_pts = pcds[i].select_by_index(np.where(mask)[0])
            dyn_pts = pcds_raw[i]
            if len(dyn_pts.points) == 0:
                continue
            T_opt = all_optimized_poses[i]
            T_opt = enforce_x_yaw(T_opt)  # 可选
            transformed = dyn_pts.transform(T_opt)
            # transformed = dyn_pts.transform(np.eye(4))
            transformed.paint_uniform_color(colors[i])
            transformed_pcds.append(transformed)
            pcd += transformed

    o3d.io.write_point_cloud('lidar_combine_optimized.pcd', pcd, print_progress=False)
    print(f"Total time: {time.time() - start_time:.2f}s")

    if verbose:
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Dynamic SLAM", 1024, 768)
        vis.show_skybox(False)
        vis.show_settings = True
        vis.set_background([0, 0, 0, 1], None)

        for i, pcd in enumerate(transformed_pcds):
            vis.add_geometry(f"pcd{i}", pcd)


        vis.setup_camera(45.0, [0, 0, 0], [10, 0, 0], [0, 0, 1])
        app.add_window(vis)
        app.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_dir', default="../../data/flow_test",
                        help='目录 *.pcd')
    args = parser.parse_args()
    # 设置随机种子
    np.random.seed(42)
    run_icp_registration(args.pcd_dir, verbose=True)