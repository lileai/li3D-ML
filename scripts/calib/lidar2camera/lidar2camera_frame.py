#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
import argparse
import glob
import os
import open3d as o3d

# ---------- 全局 ----------
T_ext = np.eye(4)  # 外参
K = np.eye(3)  # 内参
dist = np.zeros(5)  # 畸变
raw_img = None
pts = None
pts_size = 5
intensity = None
color_mode = "intensity"
alpha = 1.0

fx_real = 0.0
fy_real = 0.0
w0, h0 = 0, 0
zoom = 100
history = []  # 用于 Ctrl+Z 撤销


# ---------- 工具 ----------
def load_extrinsic(path):
    """读取 4x4 外参矩阵"""
    with open(path) as f:
        d = json.load(f)
        return np.array(d["T"]).reshape(4, 4)


def load_intrinsic(path):
    """读取 K(3x3) 和 dist(5,)"""
    with open(path) as f:
        d = json.load(f)
    K = np.array(d["K"]).reshape(3, 3)
    dist = np.array(d["dist"])
    return K, dist


def project(points, K_orig, dist_coeffs, T_l2c, mode='intensity', intensity=None):
    pts = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_cam = (T_l2c @ pts.T).T[:, :3]

    # 过滤掉深度小于 0 的点
    depth_mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[depth_mask]

    # 同时筛选 intensity
    if intensity is not None:
        intensity = intensity[depth_mask]

    # 将点投影到图像平面上
    pts2d, _ = cv2.projectPoints(pts_cam,
                                 np.zeros(3), np.zeros(3),
                                 K_orig, dist_coeffs)
    # pts2d, _ = cv2.projectPoints(pts_cam,
    #                              np.zeros(3), np.zeros(3),
    #                              K_orig, np.zeros_like(dist_coeffs))
    pts2d = pts2d.reshape(-1, 2)

    # 筛选在图像范围内的点
    mask = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w0) & \
           (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h0)
    pts2d = pts2d[mask]

    if pts2d.size == 0:
        return pts2d, np.empty((0, 3), dtype=np.uint8)

    # 根据模式生成颜色
    if intensity is not None and mode == "intensity" and intensity.shape[1] == 3:
        colors = intensity[mask] * 255
    elif intensity is not None and mode == "intensity":
        val = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        colors = cv2.applyColorMap((val * 255).astype(np.uint8), cv2.COLORMAP_AUTUMN).reshape(-1, 3)
    else:
        val = np.linalg.norm(pts_cam[mask], axis=1)
        val = (val - val.min()) / (val.max() - val.min() + 1e-8)
        colors = cv2.applyColorMap((val * 255).astype(np.uint8), cv2.COLORMAP_COOL).reshape(-1, 3)
    return pts2d, colors


def get_ship_8_points(box_xyxy, box_setting, scale, min_z, max_z):
    x1, y1, x2, y2 = box_xyxy
    x0, y0, _, _ = box_setting
    y_min = x1 / scale + y0
    x_min = y1 / scale + x0
    y_max = x2 / scale + y0
    x_max = y2 / scale + x0
    return np.array([[x_min, y_min, max_z], [x_min, y_max, max_z], [x_max, y_max, max_z], [x_max, y_min, max_z],
                     [x_min, y_min, min_z], [x_min, y_max, min_z], [x_max, y_max, min_z], [x_max, y_min, min_z]])


def project_lidar_box(lidar_box, K, dist, T_l2c):
    """
    将雷达坐标框投影到图像平面上
    :param lidar_box: 雷达坐标框的 8 个顶点坐标 (8, 3)
    :param K: 内参矩阵
    :param dist: 畸变系数
    :param T_l2c: 外参矩阵
    :return: 投影后的图像坐标框 [x1, y1, x2, y2]
    """
    # 将 8 个顶点转换为齐次坐标
    lidar_points = np.hstack([lidar_box, np.ones((8, 1))])

    # 将 3D 点转换到相机坐标系
    cam_points = (T_l2c @ lidar_points.T).T[:, :3]

    # 将 3D 点投影到图像平面上
    img_points, _ = cv2.projectPoints(cam_points, np.zeros(3), np.zeros(3), K, dist)
    img_points = img_points.reshape(-1, 2).astype(np.int32)

    return img_points


# ---------- 撤销 ----------
def pop_state():
    global T_ext, K
    if history:
        T_ext, K = history.pop()
        return True
    return False


def read_pcd_points(pcd_name):
    """优先 tensor，回落 legacy；返回 (N,3) 点云 + 可选字段"""
    # ---------- 1. 优先 tensor API ----------
    try:
        pcd = o3d.t.io.read_point_cloud(pcd_name)
    except Exception:
        pcd = None

    if pcd and "positions" in pcd.point:  # 用 in 而不用 contains
        pts = pcd.point["positions"].numpy()
        if "intensity" in pcd.point:
            return pts, pcd.point["intensity"].numpy()

    # ---------- 2. 回落 legacy API ----------
    pcd = o3d.io.read_point_cloud(pcd_name)
    if pcd.has_points():
        pts = np.asarray(pcd.points)
        if pcd.has_colors():
            return pts, np.asarray(pcd.colors)
        return pts, None

    raise RuntimeError(f"无法读取任何点云：{pcd_name}")


# ---------- 主 ----------
def main():
    global raw_img, pts, intensity, T_ext, K, dist, color_mode, w0, h0, zoom

    dirs = r"../../../data/calib/lidar2camera/data/jiaxing_capture"
    parser = argparse.ArgumentParser()
    parser.add_argument('--prj_dir', default=dirs)
    parser.add_argument("--img", default=fr"{dirs}/mapping_img")
    parser.add_argument("--pcd", default=fr"{dirs}/.pcd")
    parser.add_argument("--box_json", default=fr"{dirs}/0e_ship_masks.json")
    parser.add_argument("--calib", default=f"{dirs}/calibration_out.json")
    parser.add_argument("--fps", type=int, default=5, help="输出视频帧率")
    parser.add_argument("--out", default=f"{dirs}/lidar2camera.avi", help="输出视频路径")
    args = parser.parse_args()

    pngs = sorted(glob.glob(f'{args.img}/*.jpg', recursive=True))
    pcds = sorted(glob.glob(f'{args.pcd}/*.pcd', recursive=True))
    if len(pngs) != len(pcds):
        raise RuntimeError("图片和点云数量不一致，请检查输入目录")

    # 读内外参
    K, dist = load_intrinsic(args.calib)
    T_ext = load_extrinsic(args.calib)

    # 初始化视频写入器
    raw_img = cv2.imread(pngs[0])
    h0, w0 = raw_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (w0, h0))

    cv2.namedWindow("lidar2camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("lidar2camera", w0, h0)

    with open(args.box_json, "r") as f:
        json_data = json.load(f)

    # 创建一个字典，用于快速查找 npydir_name 对应的 box 值
    box_dict = {item["npydir_name"]: item["ship_det_box"] for item in json_data}

    # 主循环（阻塞式）
    for idx, (img_name, pcd_name) in enumerate(zip(pngs, pcds)):
        # 提取 npydir_name
        npydir_name = os.path.basename(pcd_name).split('_')[0]

        # 查找对应的 box 值
        if npydir_name in box_dict:
            ship_det_box = box_dict[npydir_name]
        else:
            print(f"未找到对应的 box 值: {npydir_name}")
            continue

        raw_img = cv2.imread(img_name)
        pts, intensity = read_pcd_points(pcd_name)

        # 投影
        pts2d, colors = project(pts, K, dist, T_ext, color_mode, intensity)

        # 绘制
        vis = raw_img.copy()
        for (x, y), col in zip(pts2d.astype(int), colors):
            col = col.tolist()
            col = [int(c * alpha) for c in col]
            cv2.circle(vis, (x, y), pts_size, col, -1)

        # 绘制 ship_det_box
        lidar_box = get_ship_8_points(np.array(ship_det_box), box_setting=[-160, 0, 40, 200], scale=5, max_z=-14.25, min_z=-20.25)
        pixel_box = project_lidar_box(lidar_box, K, dist, T_ext)

        # 绘制 3D 矩形框的边
        lines = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 侧面
        ]
        for line in lines:
            pt1 = tuple(pixel_box[line[0]])
            pt2 = tuple(pixel_box[line[1]])
            # 确保坐标是整数
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

        # 计算顶面和底面宽的中点
        mid_points = [
            (pixel_box[0] + pixel_box[1]) // 2,  # 顶面左前上和右前上的中点
            (pixel_box[2] + pixel_box[3]) // 2,  # 顶面右后上和左后上的中点
            (pixel_box[4] + pixel_box[5]) // 2,  # 底面左前下和右前下的中点
            (pixel_box[6] + pixel_box[7]) // 2   # 底面右后下和左后下的中点
        ]

        # 绘制中点框
        mid_lines = [
            (0, 1), (1, 3), (3, 2), (2, 0)
        ]
        for line in mid_lines:
            pt1 = tuple(mid_points[line[0]])
            pt2 = tuple(mid_points[line[1]])
            # 确保坐标是整数
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(vis, pt1, pt2, (0, 0, 255), 2)

        x_min, y_min = np.min(mid_points, axis=0)
        x_max, y_max = np.max(mid_points, axis=0)

        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        vis = cv2.addWeighted(vis, alpha, raw_img, 1 - alpha, 0)

        cv2.imshow("lidar2camera", vis)
        key = cv2.waitKey(0) & 0xFF  # 阻塞等待按键
        if key == 27:  # Esc 提前退出
            break
        elif key == ord('z'):  # Ctrl+Z 撤销
            if pop_state():
                print("Undo")
            else:
                print("Nothing to undo")
            # 撤销后重新投影显示当前帧
            continue

        # 写视频
        vw.write(vis)
        print(f"Frame {idx + 1}/{len(pngs)} written")

    vw.release()
    cv2.destroyAllWindows()
    print(f"Done! Video saved to {args.out}")


if __name__ == "__main__":
    main()
