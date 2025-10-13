#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简手动激光雷达到激光雷达外参标定工具
python lidar2lidar_keyonly.py target.pcd source.pcd init.json_files

按键：
  Q/A ——  绕 X 轴 ±
  W/S ——  绕 Y 轴 ±
  E/D ——  绕 Z 轴 ±
  R/F ——  沿 X 轴 ±
  T/G ——  沿 Y 轴 ±
  Y/H ——  沿 Z 轴 ±
  Z   ——  重置
  X   ——  保存外参
"""

import json
import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from open3d.visualization.gui import KeyEvent, Widget
import open3d.visualization.rendering as rendering

DEFAULT_STEP_R = 0.3          # deg
DEFAULT_STEP_T = 0.06         # m
COLOR_TARGET = [0.8, 0.8, 0.8]
COLOR_SOURCE = [1.0, 0.2, 0.2]

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise RuntimeError(f"Cannot read: {path}")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data.get("extrinsic", data), dtype=float)

def save_json(mat, idx, data_dir):
    # 将外参矩阵转换为字典格式
    extrinsic_dict = {
        "extrinsic": mat.tolist(),
    }

    # 保存为 JSON 文件
    with open(rf"{data_dir}/lidar2lidar_extrinsic_{idx}.json_files", "w") as f:
        json.dump(extrinsic_dict, f, indent=4)

    print(f"Saved lidar2lidar_extrinsic_{idx}.json_files")

class LidarKeyOnlyApp:
    def __init__(self, tgt_path, src_path, init_path, data_dir):
        self.pcd_tgt = load_pcd(tgt_path)
        self.pcd_src = load_pcd(src_path)
        self.pcd_src_backup = self.pcd_src  # keep original
        self.T = load_json(init_path)
        self.T_init = self.T.copy()
        self.data_dir = data_dir

        self.step_r = np.deg2rad(DEFAULT_STEP_R)  # 初始旋转步长
        self.step_t = DEFAULT_STEP_T  # 初始平移步长
        self.save_idx = 0

        # 显示状态标志
        self.show_tgt = True
        self.show_src = True
        self.show_tgt_frame = True
        self.show_src_frame = True

        # 操作历史记录
        self.history = [self.T.copy()]
        self.history_idx = 0

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "lidar2lidar keyonly", 1200, 800)

        self.widget = gui.SceneWidget()
        self.widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer)
        self.window.add_child(self.widget)

        # 添加坐标原点
        self.add_coordinate_frame()

        # 添加目标点云
        self.widget.scene.add_geometry("tgt", self.pcd_tgt,
                                       o3d.visualization.rendering.MaterialRecord())
        self.update_source()

        bounds = self.pcd_tgt.get_axis_aligned_bounding_box()
        self.widget.setup_camera(60, bounds, bounds.get_center())

        # 绑定到窗口
        self.window.set_on_key(self.on_key)

        # 计算 z 最大的点
        max_z_point = np.array([bounds.get_max_bound()[0], bounds.get_max_bound()[1], bounds.get_max_bound()[2]])
        self.pose_label = self.widget.add_3d_label(max_z_point + np.array([0, 0, 1.5]), "Pose: [0.000, 0.000, 0.000] r=0.00° p=0.00° y=0.00°")
        self.step_label = self.widget.add_3d_label(max_z_point + np.array([0, 0, 1.0]), "Rotation step: 0.30°\nTranslation step: 0.060m")

        # 初始更新
        self.update_pose_label()
        self.update_step_label()

    def add_coordinate_frame(self):
        # 创建 target 点云的坐标原点
        self.tgt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        self.widget.scene.add_geometry("tgt_frame", self.tgt_frame,
                                       o3d.visualization.rendering.MaterialRecord())

    def update_source(self):
        # taipu_main_side. 重新从备份复制一份干净点云
        tmp = self.pcd_src_backup + o3d.geometry.PointCloud()
        # 2. 一次性施加当前完整变换
        tmp.transform(self.T)
        # 3. 替换场景里的几何体
        self.widget.scene.remove_geometry("src")
        tmp.paint_uniform_color([1, 0, 0])
        self.widget.scene.add_geometry("src", tmp,
                                       o3d.visualization.rendering.MaterialRecord())

    def update_pose_label(self):
        t = self.T[:3, 3]
        roll, pitch, yaw = self.rot2euler(self.T[:3, :3])
        pose_text = (f"Pose: [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]  "
                     f"r={np.rad2deg(roll):+.2f}°  "
                     f"p={np.rad2deg(pitch):+.2f}°  "
                     f"y={np.rad2deg(yaw):+.2f}°")
        self.pose_label.text = pose_text

    def update_step_label(self):
        step_text = (f"Rotation step: {np.rad2deg(self.step_r):+.2f}°\n"
                     f"Translation step: {self.step_t:.3f}m")
        self.step_label.text = step_text

    def print_pose(self):
        t = self.T[:3, 3]
        roll, pitch, yaw = self.rot2euler(self.T[:3, :3])
        print(f"[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]  "
              f"r={np.rad2deg(roll):+.2f}°  "
              f"p={np.rad2deg(pitch):+.2f}°  "
              f"y={np.rad2deg(yaw):+.2f}°")

    def print_step(self):
        print(f"Rotation step: {np.rad2deg(self.step_r):+.2f}°")
        print(f"Translation step: {self.step_t:.3f}m")

    def rot2euler(self, R):
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        return (np.arctan2(R[2, 1], R[2, 2]),
                np.arctan2(-R[2, 0], max(sy, 1e-6)),
                np.arctan2(R[1, 0], R[0, 0]))

    def on_key(self, event):
        if event.type != KeyEvent.DOWN:
            return int(Widget.EventCallbackResult.IGNORED)

        key = chr(event.key).upper()
        dR = np.eye(3)  # 默认值，确保 dR 总是有定义

        if key == 'Q':
            dR = o3d.geometry.get_rotation_matrix_from_xyz([self.step_r, 0, 0])
        elif key == 'A':
            dR = o3d.geometry.get_rotation_matrix_from_xyz([-self.step_r, 0, 0])
        elif key == 'W':
            dR = o3d.geometry.get_rotation_matrix_from_xyz([0, self.step_r, 0])
        elif key == 'S':
            dR = o3d.geometry.get_rotation_matrix_from_xyz([0, -self.step_r, 0])
        elif key == 'E':
            dR = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, self.step_r])
        elif key == 'D':
            dR = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, -self.step_r])
        elif key == 'R':
            self.T[:3, 3] += [self.step_t, 0, 0]
        elif key == 'F':
            self.T[:3, 3] += [-self.step_t, 0, 0]
        elif key == 'T':
            self.T[:3, 3] += [0, self.step_t, 0]
        elif key == 'G':
            self.T[:3, 3] += [0, -self.step_t, 0]
        elif key == 'Y':
            self.T[:3, 3] += [0, 0, self.step_t]
        elif key == 'H':
            self.T[:3, 3] += [0, 0, -self.step_t]
        elif key == 'Z':
            self.T = self.T_init.copy()
        elif key == 'X':
            save_json(self.T, self.save_idx, self.data_dir)  # 修改为调用 save_json
            self.save_idx += 1
            return int(Widget.EventCallbackResult.CONSUMED)
        elif key == 'U':  # 增加旋转步长
            self.step_r += np.deg2rad(0.1)
            print(f"Rotation step increased to {np.rad2deg(self.step_r):.2f}°")
        elif key == 'J':  # 减少旋转步长
            self.step_r -= np.deg2rad(0.1)
            self.step_r = max(self.step_r, np.deg2rad(0.1))  # 防止步长变为负值
            print(f"Rotation step decreased to {np.rad2deg(self.step_r):.2f}°")
        elif key == 'I':  # 增加平移步长
            self.step_t += 0.001
            print(f"Translation step increased to {self.step_t:.3f}m")
        elif key == 'K':  # 减少平移步长
            self.step_t -= 0.001
            self.step_t = max(self.step_t, 0.001)  # 防止步长变为负值
            print(f"Translation step decreased to {self.step_t:.3f}m")
        elif key == 'N':  # 回退操作
            if self.history_idx > 0:
                self.history_idx -= 1
                self.T = self.history[self.history_idx].copy()
                print(f"Undo operation. Current state: {self.history_idx + 1}/{len(self.history)}")
            else:
                print("No more operations to undo.")
        elif key == 'taipu_main_side':  # 切换显示/隐藏 target 点云
            self.show_tgt = not self.show_tgt
            if self.show_tgt:
                self.widget.scene.add_geometry("tgt", self.pcd_tgt,
                                               o3d.visualization.rendering.MaterialRecord())
            else:
                self.widget.scene.remove_geometry("tgt")
        else:
            return int(Widget.EventCallbackResult.IGNORED)

        # 旋转部分
        dT = np.eye(4)
        dT[:3, :3] = dR
        self.T = self.T @ dT

        # 保存当前状态到历史记录
        if key != 'N' and key != 'Z':
            self.history = self.history[:self.history_idx + 1]
            self.history.append(self.T.copy())
            self.history_idx += 1

        print(f"[Key] {key}")
        self.update_source()
        if key in 'QWERTYASDFGH':
            self.print_pose()
        elif key in 'UIJK':
            self.print_step()

        # 更新标签
        self.update_pose_label()
        self.update_step_label()
        return int(Widget.EventCallbackResult.CONSUMED)

    def run(self):
        gui.Application.instance.run()
# ---------- main ----------
def main():
    data_path = r"../../../data/calib/lidar2lidar/data"
    parser = argparse.ArgumentParser(description="手动激光雷达到激光雷达外参标定工具")
    parser.add_argument("--data_dir", default=data_path, help="数据路径")
    parser.add_argument("--target", default=rf"{data_path}/pcd_1_1755848727.1383927.pcd", help="目标点云文件 (target.pcd)")
    parser.add_argument("--source", default=rf"{data_path}/pcd_2_1755848727.1383927.pcd", help="源点云文件 (source.pcd)")
    parser.add_argument("--init", default=rf"{data_path}/init.json", help="初始外参 JSON 文件 (init.json)")
    args = parser.parse_args()
    app = LidarKeyOnlyApp(args.target, args.source, args.init, args.data_dir)
    app.run()

if __name__ == "__main__":
    main()