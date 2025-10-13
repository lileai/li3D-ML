#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
draw_frames.py
读取一帧点云 + 外参矩阵，可视化雷达坐标系与相机坐标系
"""
import open3d as o3d
import numpy as np
import json
import argparse
import open3d.visualization.gui as gui

def load_extrinsic(path):
    with open(path) as f:
        d = json.load(f)
        return d["T"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", default=r"D:\program\li3D-ML\scripts\calib\lidar2camera\data\qingshan\rename_2025_05_29_10_08_44_524.pcd", help="点云文件 .pcd")
    parser.add_argument("--extr", default=r"D:\program\li3D-ML\scripts\calib\lidar2camera\data\qingshan\calibration_out.json", help="外参 4×4 JSON")
    args = parser.parse_args()

    # 1. 读取点云
    pcd = o3d.io.read_point_cloud(args.pcd)
    print(f"Loaded {len(pcd.points)} points")

    # 2. 读取外参
    T_ext = load_extrinsic(args.extr)          # 4×4
    print("Extrinsic matrix:\n", T_ext)
    print("inv Extrinsic matrix:\n", np.linalg.inv(T_ext))

    # 3. 创建坐标系
    frame_lidar = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])            # 雷达系（RGB）

    frame_cam = o3d.geometry.TriangleMesh.create_coordinate_frame() + frame_lidar
    frame_cam.transform(np.linalg.inv(T_ext))  # 就地变换到相机系

    # 4. 可视化
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(
        width=1660, height=900)
    vis.show_skybox(False)
    vis.show_settings = True
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所
    vis.add_geometry("pcd", pcd)
    vis.add_geometry("frame_lidar", frame_lidar)
    vis.add_geometry("frame_cam", frame_cam)

    app.add_window(vis)
    app.run()

if __name__ == "__main__":
    main()