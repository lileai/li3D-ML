#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import open3d as o3d
import json
import open3d.visualization.gui as gui

# ---------- 工具 ----------
def load_calib(json_path):
    with open(json_path, 'r') as f:
        c = json.load(f)
    K = np.array(c['K']).reshape(3, 3)
    D = np.array(c['dist'])
    T = np.array(c['T']).reshape(4, 4)
    return K, D, T

def undistort_img(img, K, D):
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), 5)
    img_ud = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return img_ud, newK

def colorize_cloud(pcd_np, img, K, T):
    """给点云上色"""
    h, w = img.shape[:2]

    # 1. lidar → camera 坐标
    pts_cam = (T[:3, :3] @ pcd_np.T + T[:3, 3:4]).T  # (N,3)

    # 2. 投影到像素
    uv = K @ pts_cam.T
    uv = uv[:2] / uv[2]                                # (2,N)
    u, v = uv[0], uv[1]

    # 3. 筛选有效投影
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = u[mask].astype(int), v[mask].astype(int)

    # 4. 取颜色
    rgb = img[v, u]                                    # (N,3)  BGR
    rgb = rgb[:, ::-1] / 255.0                        # 转 RGB 0~1

    # 5. 构建新点云
    cloud_colored = o3d.geometry.PointCloud()
    cloud_colored.points = o3d.utility.Vector3dVector(pcd_np[mask])
    cloud_colored.colors = o3d.utility.Vector3dVector(rgb)
    return cloud_colored

# ---------- 主 ----------
if __name__ == '__main__':
    data_path = r"../../../data/calib/lidar2camera/data/taipu_side"
    img_path   = fr'{data_path}/mapping_2025_09_03_14_24_54_470594_img.jpg'
    pcd_path   = fr'{data_path}/rename_2025_09_03_14_24_54_000000.pcd'
    calib_path = fr'{data_path}/calibration_out.json'

    # 1. 读取数据
    img   = cv2.imread(img_path)
    pcd   = o3d.io.read_point_cloud(pcd_path)
    K, D, T = load_calib(calib_path)

    # 2. 图像去畸变
    img_ud, K_new = undistort_img(img, K, D)

    # 3. 上色
    cloud_colored = colorize_cloud(np.asarray(pcd.points), img_ud, K_new, T)

    # 4. 保存 / 可视化
    o3d.io.write_point_cloud(f'{data_path}/cloud_colored.pcd', cloud_colored)
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(
        width=1660, height=900)
    vis.show_skybox(False)
    vis.show_settings = True
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所
    vis.add_geometry("cloud_colored", cloud_colored)


    app.add_window(vis)
    app.run()