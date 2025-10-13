#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
船舶 SLAM – 优化版
python ship_slam.py --dir ./data --loop taipu_main_side --voxel 0.taipu_main_side --step 3
"""
import os, glob, time, argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import open3d.visualization.gui as gui
import matplotlib.pyplot as plt

# ========== 可调参数 ==========
VOXEL       = 0.10     # 体素
STEP        = 3        # 每 STEP 帧取 taipu_main_side 关键帧
K_FPFH      = 0.15     # FPFH 半径
K_RANSAC    = 0.30     # RANSAC 最大距离
K_ICP       = 0.15     # ICP 最大距离
LOOP_DIST   = 2.0      # 回环空间距离
LOOP_TIME   = 30       # 回环时间间隔（帧数）
# ==============================

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
    pcd = pcd.voxel_down_sample(voxel_size=0.2)
    normals = estimate_normals_vec(np.asarray(pcd.points), radius=VOXEL * 2, max_nn=30)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def extract_dynamic(prev, curr):
    """体素差分提取动态点"""
    prev_down = prev.voxel_down_sample(VOXEL*0.5)
    curr_down = curr.voxel_down_sample(VOXEL*0.5)
    tree = cKDTree(np.asarray(prev_down.points))
    dist, _ = tree.query(np.asarray(curr_down.points))
    mask = dist > VOXEL*1.5
    return curr_down.select_by_index(np.where(mask)[0]) if mask.any() else curr_down

def register_fpfh_icp(src, dst):
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=K_FPFH, max_nn=100))
    dst_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        dst, o3d.geometry.KDTreeSearchParamHybrid(radius=K_FPFH, max_nn=100))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, src_fpfh, dst_fpfh,
        False,                     # 新增：mutual_filter
        K_RANSAC,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(K_RANSAC)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1.0))

    result = o3d.pipelines.registration.registration_icp(
        src, dst, K_ICP, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def build_pose_graph(kf_list):
    N = len(kf_list)
    pg = o3d.pipelines.registration.PoseGraph()
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    T_acc = np.eye(4)

    for i in range(1, N):
        reg = register_fpfh_icp(kf_list[i], kf_list[i-1])
        T_acc = reg.transformation @ T_acc
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(T_acc)))
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            kf_list[i-1], kf_list[i], K_ICP, reg.transformation)
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i-1, i, reg.transformation, info, uncertain=False))
    return pg

def add_loop_closures(pg, kf_list):
    centers = np.array([k.get_center() for k in kf_list])
    tree = cKDTree(centers)
    N = len(kf_list)
    added = 0
    for i in range(N):
        idxs = tree.query_ball_point(centers[i], LOOP_DIST)
        for j in idxs:
            if abs(i-j) < LOOP_TIME:
                continue
            reg = register_fpfh_icp(kf_list[i], kf_list[j])
            if reg.fitness < 0.25:
                continue
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                kf_list[j], kf_list[i], K_ICP, reg.transformation)
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                j, i, reg.transformation, info, uncertain=True))
            added += 1
    print(f"添加回环边: {added}")

def optimize(pg):
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=K_ICP))

def fuse(kf_list, pg):
    whole = o3d.geometry.PointCloud()
    colors = plt.cm.jet(np.linspace(0, 1, len(kf_list)))[:, :3]
    for i, pcd in enumerate(kf_list):
        pcd.transform(np.linalg.inv(pg.nodes[i].pose))
        pcd.paint_uniform_color(colors[i])
        whole += pcd
    whole = whole.voxel_down_sample(VOXEL*0.8)
    o3d.io.write_point_cloud("ship_whole.pcd", whole)
    return whole

def main(pcd_dir, loop):
    files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    assert len(files) > 1
    print(f"载入 {len(files)} 帧，步长 {STEP}")
    pcds = [load_pcd(f) for f in files[::STEP]]

    t0 = time.time()
    pg = build_pose_graph(pcds)
    if loop:
        add_loop_closures(pg, pcds)
    optimize(pg)
    ship = fuse(pcds, pg)
    print(f"耗时 {time.time()-t0:.1f}s，输出 ship_whole.pcd")
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
    vis.add_geometry("SHIP", ship)
    # vis.add_geometry("box", box)
    vis.setup_camera(
        45.0,  # fov
        np.array([0, 0, 0], dtype=np.float32),  # center
        np.array([10, 0, 0], dtype=np.float32),  # eye
        np.array([0, 0, 1], dtype=np.float32)  # up
    )
    app.add_window(vis)
    app.run()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="../../data/flow_data", help="PCD 目录")
    parser.add_argument("--loop", type=int, default=1, help="是否回环")
    parser.add_argument("--voxel", type=float, default=0.10, help="体素")
    args = parser.parse_args()
    VOXEL = args.voxel
    main(args.dir, bool(args.loop))