#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lidar2camera_undo_pnp.py
手动微调 + 自动初值（PnP 选点）
用法：
  1. 运行脚本
  2. 按 m 进入选点模式
  3. 左键点击图像 → 终端输入对应 3D 坐标（x y z 空格分隔）
  4. ≥6 组后按 p 自动计算外参初值
  5. 用原有按键/滑块微调
  6. 按 ESC 退出
"""
import cv2
import numpy as np
import json
import argparse
import os
import open3d as o3d

# ---------- 全局 ----------
T_ext = np.eye(4)          # 外参
K = np.eye(3)              # 内参
dist = np.zeros(5)         # 畸变
raw_img = None
pts = None
intensity = None
color_mode = "intensity"
alpha = 0.5
fx_real = 0.0
fy_real = 0.0
w0, h0 = 0, 0
zoom = 100
history = []               # 撤销栈

# ✅ 新增：选点模式相关
pick_mode = False
pts_2d = []                # 图像点
pts_3d = []                # 点云点

# ---------- 工具 ----------
def draw_frame(img, orig_2d, axes_2d, len_px=60, thickness=2):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for ax2d, c in zip(axes_2d, colors):
        end = orig_2d + len_px * ax2d
        pt1 = tuple(orig_2d.astype(np.int32))  # 确保起点是整数
        pt2 = tuple(end.astype(np.int32))  # 确保终点是整数

        # 检查坐标是否在图像范围内
        if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
            0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
            cv2.arrowedLine(img, pt1, pt2, color=c, thickness=thickness, tipLength=0.2)


def load_extrinsic(path):
    with open(path) as f:
        return np.array(json.load(f))

def load_intrinsic(path):
    with open(path) as f:
        d = json.load(f)
    K = np.array(d["K"]).reshape(3, 3)
    dist = np.array(d["dist"])
    return K, dist

def make_mods(deg, trans):
    mods = []
    for i in range(12):
        tf = [0]*6
        tf[i//2] = 1 if i % 2 == 0 else -1
        ax, ay, az = [t*deg*np.pi/180 for t in tf[:3]]
        rot, _ = cv2.Rodrigues(np.array([ax, ay, az]))
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = np.array(tf[3:])*trans
        mods.append(T)
    return mods

def project(points, K_orig, dist_coeffs, T_l2c, mode='intensity', intensity=None):
    pts = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_cam = (T_l2c @ pts.T).T[:, :3]
    depth_mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[depth_mask]
    if intensity is not None:
        intensity = intensity[depth_mask]
    pts2d, _ = cv2.projectPoints(pts_cam,
                                 np.zeros(3), np.zeros(3),
                                 K_orig, dist_coeffs)
    # pts2d, _ = cv2.projectPoints(pts_cam,
    #                              np.zeros(3), np.zeros(3),
    #                              K_orig, np.zeros_like(dist_coeffs))
    pts2d = pts2d.reshape(-1, 2)
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

def save_result(T, K_orig, prj_dir):
    out = {"T": T.tolist(), "K": K_orig.tolist(), "dist": dist.flatten().tolist()}
    with open(f"{prj_dir}/calibration_out.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved!")

# ---------- 撤销 ----------
def push_state():
    history.append((T_ext.copy(), K.copy()))

def pop_state():
    global T_ext, K
    if history:
        T_ext, K = history.pop()
        return True
    return False

# ✅ 新增：鼠标回调
def mouse_click(event, x, y, flags, param):
    global pts_2d, pts_3d, pick_mode, last_action
    if not pick_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_2d.append([x, y])
        print(f"[pick] 2D #{len(pts_2d)}:  {x, y}")
        xyz_str = input("[pick] 输入 3D x y z（空格）: ").strip()
        try:
            xyz = list(map(float, xyz_str.split()))
            if len(xyz) != 3:
                raise ValueError
            pts_3d.append(xyz)
            print(f"[pick] 3D #{len(pts_3d)}:  {xyz}")
        except Exception:
            print("[pick] 非法输入，已忽略")
            pts_2d.pop()

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
    global raw_img, pts, intensity, T_ext, K, dist, color_mode, w0, h0, zoom, alpha, pick_mode, pts_2d, pts_3d, last_action
    dirs = r"../../../data/calib/lidar2camera/data/jiaxing_capture"
    parser = argparse.ArgumentParser()
    parser.add_argument('--prj_dir', default=dirs)
    parser.add_argument("--img",  default=r"D:\program\li3D-ML\data\calib\lidar2camera\data\jiaxing_capture\mapping_img\mapping_2025_11_04_12_04_55_863299_img.jpg")
    parser.add_argument("--pcd",  default=r"D:\program\li3D-ML\data\calib\lidar2camera\data\jiaxing_capture\.pcd\1762229095.8632991_MERGE.pcd")
    parser.add_argument("--intr", default=fr"{dirs}/intrinsic.json")
    parser.add_argument("--extr", default=fr"{dirs}/extrinsic.json")
    args = parser.parse_args()
    np.random.seed(42)

    K, dist = load_intrinsic(args.intr)
    K_orig = K.copy()
    T_ext = load_extrinsic(args.extr)

    raw_img = cv2.imread(args.img)
    if raw_img is None:
        raise FileNotFoundError(args.img)
    h0, w0 = raw_img.shape[:2]

    pts, intensity = read_pcd_points(args.pcd)

    history.clear()
    push_state()

    cv2.namedWindow("lidar2camera", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("lidar2camera")
    cv2.resizeWindow("lidar2camera", w0, h0 + 100)
    cv2.setMouseCallback("lidar2camera", mouse_click)

    cv2.createTrackbar("deg_step(x100)", "lidar2camera", 30, 500, lambda x: None)
    cv2.createTrackbar("trans_step(cm)", "lidar2camera", 1000, 1000, lambda x: None)
    cv2.createTrackbar("zoom(%)", "lidar2camera", 100, 400, lambda x: None)
    cv2.createTrackbar("point_size", "lidar2camera", 1, 20, lambda x: None)
    cv2.createTrackbar("toggle_mode", "lidar2camera", 0, 1, lambda x: None)
    cv2.createTrackbar("alpha", "lidar2camera", int(alpha * 100), 100, lambda x: None)

    key_desc = {ord('q'): "+X°", ord('a'): "-X°", ord('w'): "+Y°", ord('s'): "-Y°",
                ord('e'): "+Z°", ord('d'): "-Z°", ord('r'): "+XT", ord('f'): "-XT",
                ord('t'): "+YT", ord('g'): "-YT", ord('y'): "+ZT", ord('h'): "-ZT"}

    T0 = T_ext.copy()
    last_action = ""

    while cv2.getWindowProperty("lidar2camera", cv2.WND_PROP_VISIBLE) >= 1:
        deg   = cv2.getTrackbarPos("deg_step(x100)", "lidar2camera") / 100.0
        trans = cv2.getTrackbarPos("trans_step(cm)", "lidar2camera") / 100.0
        zoom  = cv2.getTrackbarPos("zoom(%)", "lidar2camera") / 100.0
        psize = cv2.getTrackbarPos("point_size", "lidar2camera")
        alpha = cv2.getTrackbarPos("alpha", "lidar2camera") / 100.0

        new_switch = cv2.getTrackbarPos("toggle_mode", "lidar2camera")
        if new_switch != 0:
            cv2.setTrackbarPos("toggle_mode", "lidar2camera", 0)
            color_mode = "distance" if color_mode == "intensity" else "intensity"
            last_action = f"mode -> {color_mode}"

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k == ord('n'):
            if pop_state():
                last_action = "Undo"
            else:
                last_action = "Nothing to undo"
        # ✅ 选点模式开关
        elif k == ord('m'):
            pick_mode = not pick_mode
            last_action = f"Pick mode {'ON' if pick_mode else 'OFF'}"
        # ✅ PnP 计算初值
        elif k == ord('p'):
            if len(pts_2d) >= 6:
                obj = np.array(pts_3d, dtype=np.float32)
                img = np.array(pts_2d, dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(obj, img, K_orig, dist)
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    T_ext = np.eye(4)
                    T_ext[:3, :3] = R
                    T_ext[:3, 3] = tvec.flatten()
                    history.clear()
                    push_state()

                    # ========== 输出初值矩阵 ==========
                    print("\n[PNP] 外参初值 T_ext (4×4):")
                    print(np.array2string(T_ext, separator=', '))
                    # 如需保存
                    np.savetxt("pnp_T_init.txt", T_ext)
                    print("[PNP] 已保存 pnp_T_init.txt\n")
                    # ==================================

                    last_action = f"PnP init done ({len(pts_2d)} pts)"
                else:
                    last_action = "PnP failed"
            else:
                last_action = f"Need >=6 pts (now {len(pts_2d)})"
        elif k in key_desc:
            idx = list(key_desc.keys()).index(k)
            T_ext = T_ext @ make_mods(deg, trans)[idx]
            push_state()
            last_action = key_desc[k]
        elif k in (ord('u'), ord('j')):
            K[0, 0] *= 1.005 if k == ord('u') else 1/1.005
            push_state()
            last_action = f"fx={K[0,0]:.1f}"
        elif k in (ord('i'), ord('k')):
            K[1, 1] *= 1.005 if k == ord('i') else 1/1.005
            push_state()
            last_action = f"fy={K[1,1]:.1f}"
        elif k == ord('o'):  # 增加 k1
            dist[0][0] += 0.001
            last_action = f"Adjusted k1: {float(dist[0][0]):.4f}"
        elif k == ord('l'):  # 减少 k1
            dist[0][0] -= 0.001
            last_action = f"Adjusted k1: {float(dist[0][0]):.4f}"
        elif k == ord('z'):
            T_ext = T0.copy()
            K, dist = load_intrinsic(args.intr)
            history.clear()
            push_state()
            last_action = "Reset"
        elif k == ord('x'):
            save_result(T_ext, K, args.prj_dir)
            last_action = "Saved"

        cur_K_img = K.copy()
        pts2d, colors = project(pts, cur_K_img, dist, T_ext, color_mode, intensity)
        vis = raw_img.copy()
        for (x, y), col in zip(pts2d.astype(int), colors):
            col = [int(c * alpha) for c in col]
            cv2.circle(vis, (x, y), psize, col, -1)

        center_w = np.mean(pts, axis=0)
        axes_w   = np.eye(3)
        pts_center_axes = np.vstack([center_w, center_w + axes_w])
        pts_cam = (T_ext @ np.hstack([pts_center_axes, np.ones((4, 1))]).T).T[:, :3]
        if np.all(pts_cam[:, 2] > 0):
            pts2d_center_axes, _ = cv2.projectPoints(
                pts_cam, np.zeros(3), np.zeros(3), cur_K_img, dist)
            pts2d_center_axes = pts2d_center_axes.reshape(-1, 2)
            orig_2d = pts2d_center_axes[0]
            R_2d    = pts2d_center_axes[1:] - orig_2d
            draw_frame(vis, orig_2d, R_2d, len_px=10, thickness=2)

        T_w2c = T_ext
        T_c2w = np.linalg.inv(T_w2c)
        center_w = T_c2w[:3, 3]
        axes_w   = T_c2w[:3, :3]
        pts_cam_axes = np.vstack([center_w, center_w + axes_w])
        pts_cam = (T_w2c @ np.hstack([pts_cam_axes, np.ones((4, 1))]).T).T[:, :3]
        if np.all(pts_cam[:, 2] > 0):
            pts2d_cam_axes, _ = cv2.projectPoints(
                pts_cam, np.zeros(3), np.zeros(3), cur_K_img, dist)
            pts2d_cam_axes = pts2d_cam_axes.reshape(-1, 2)
            orig_2d = pts2d_cam_axes[0]
            axes_2d = pts2d_cam_axes[1:] - orig_2d
            draw_frame(vis, orig_2d, axes_2d, len_px=10, thickness=1)

        vis = cv2.addWeighted(vis, alpha, raw_img, 1 - alpha, 0)
        if abs(zoom - 1.0) > 0.01:
            vis = cv2.resize(vis, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

        # ✅ 选点模式提示
        if pick_mode:
            cv2.putText(vis, ">>> PICK MODE: click img -> input xyz -> press 'p' <<<",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        t_vec = T_ext[:3, 3]
        info = [f"zoom={zoom*100:.0f}%", f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}", f"alpha={alpha:.2f}", f"trans(m) X={t_vec[0]:+.2f} Y={t_vec[1]:+.2f} Z={t_vec[2]:+.2f}"]
        if last_action:
            info.append(last_action)
        for i, txt in enumerate(info):
            cv2.putText(vis, txt, (10, 30 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("lidar2camera", vis)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()