#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_horizon_panel.py
地平线按键步幅调畸变 + 顶部信息面板（文字不遮挡画面）
"""
import cv2
import numpy as np
import json
import argparse
from scipy.optimize import least_squares

PANEL_H = 0  # 顶部信息面板高度
STEP_MAP = {'k1': 0, 'k2': 1, 'k3': 4, 'p1': 2, 'p2': 3}

img_orig = None
pts_horizon = []
K = np.eye(3)
dist = np.zeros(5)
history = []
step = 0.01
new_K = None
mapx = mapy = None
new_w = new_h = 0


def push_state():
    history.append((K.copy(), dist.copy()))


def pop_state():
    global K, dist
    if history:
        K, dist = history.pop()
        return True
    return False


def save_result(path, img_size):
    out = {"K": K.tolist(), "dist": dist.tolist(), "img_size": img_size}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved ->", path)


def horizon_error(pts2d):
    if len(pts2d) < 2:
        return 0.0
    pts = np.array(pts2d, dtype=np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    a, b, c = vy, -vx, vx * y0 - vy * x0
    dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / np.hypot(a, b)
    return np.mean(dists)


def build_remap():
    global mapx, mapy, new_K, new_w, new_h
    h, w = img_orig.shape[:2]
    new_w, new_h = w, h

    # 1. 用最新焦距生成【新的】输出内参矩阵
    new_K = np.array([[K[0, 0], 0, w / 2],
                      [0, K[1, 1], h / 2],
                      [0, 0, 1]], dtype=float)

    # 2. 强制与输入 K 不同对象，再传入
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (new_w, new_h), cv2.CV_32FC1)


def undistort_and_draw():
    img = cv2.remap(img_orig, mapx, mapy, cv2.INTER_LINEAR)
    if len(pts_horizon) > 1:
        pts_ud = cv2.undistortPoints(
            np.array(pts_horizon, dtype=np.float32).reshape(-1, 1, 2),
            K, dist, P=new_K).reshape(-1, 2)
        for x, y in pts_ud:
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
        if len(pts_ud) >= 2:
            [vx, vy, x0, y0] = cv2.fitLine(pts_ud, cv2.DIST_L2, 0, 0.01, 0.01)
            pts_line = np.array([[x0 + vx * t, y0 + vy * t] for t in np.linspace(-2000, 2000, 200)], dtype=int)
            cv2.polylines(img, [pts_line], False, (0, 255, 0), 0)
    return img


def mouse(event, x, y, flags, param):
    # 直接原图坐标（remap 后分辨率就是 new_w/new_h）
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_horizon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(pts_horizon) > 1:
        pts_horizon.append(pts_horizon[0])


def auto_calc():
    if len(pts_horizon) < 3:
        return

    pts_norm = np.array(pts_horizon, dtype=np.float32)  # 原始点击（大图坐标）

    # 1. 粗搜 k1、k2  25×21 = 525 次
    best_err, best_k1, best_k2 = 1e9, 0., 0.
    for k1 in np.linspace(-0.6, 0.6, 25):
        for k2 in np.linspace(-0.15, 0.15, 21):
            dist_tmp = np.array([k1, k2, 0, 0, 0])
            pts_ud = cv2.undistortPoints(pts_norm.reshape(-1, 1, 2), K, dist_tmp, P=K).reshape(-1, 2)
            err = horizon_error(pts_ud)
            if err < best_err:
                best_err, best_k1, best_k2 = err, k1, k2

    # 2. 以粗搜结果作为初值， refine 全部 7 参数
    x0 = np.array([best_k1, best_k2, 0., 0., 0., K[0, 0], K[1, 1]])

    def cost(x):
        d = np.array([x[0], x[1], x[2], x[3], x[4]])
        k_mat = K.copy()
        k_mat[0, 0], k_mat[1, 1] = x[5], x[6]
        pts_ud = cv2.undistortPoints(pts_norm.reshape(-1, 1, 2), k_mat, d, P=k_mat).reshape(-1, 2)
        # 返回每条点到拟合直线的距离
        if len(pts_ud) < 2:
            return np.ones(len(pts_ud)) * 1e3
        [vx, vy, x0, y0] = cv2.fitLine(pts_ud, cv2.DIST_L2, 0, 0.01, 0.01)
        a, b, c = vy, -vx, vx * y0 - vy * x0
        return np.abs(a * pts_ud[:, 0] + b * pts_ud[:, 1] + c) / np.hypot(a, b)

    res = least_squares(cost, x0, method='trf', max_nfev=1000,
                        bounds=([-0.8, -0.2, -0.02, -0.02, -0.05, K[0, 0] * 0.7, K[1, 1] * 0.7],
                                [0.8, 0.2, 0.02, 0.02, 0.05, K[0, 0] * 1.3, K[1, 1] * 1.3]))
    k1, k2, p1, p2, k3, fx, fy = res.x
    dist[0], dist[1], dist[2], dist[3], dist[4] = k1, k2, p1, p2, k3
    K[0, 0], K[1, 1] = fx, fy
    push_state()
    build_remap()
    print(f"auto fast: k1={k1:.4f} k2={k2:.4f} k3={k3:.4f} "
          f"p1={p1:.4f} p2={p2:.4f} fx={fx:.1f} fy={fy:.1f} → err={res.cost:.3f}px")


def main():
    global img_orig, K, dist, history, pts_horizon, step, mapx, mapy, new_K
    data_dir = "../../../data/calib/camera_intrinsic/data"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img",
                        default=r"D:\program\li3D-ML\scripts\calib\camera_intrinsic\data\suqian_thermal\10.23.221.151_02_20251013012729872.jpeg")
    parser.add_argument("-o", "--out", default=fr"{data_dir}/suqian_thermal/intrinsic.json")
    args = parser.parse_args()

    img_orig = cv2.imread(args.img)
    if img_orig is None:
        raise FileNotFoundError(args.img)
    h, w = img_orig.shape[:2]

    f0 = max(w, h) * 0.8
    K = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], dtype=float)
    dist = np.zeros(5)
    push_state()
    build_remap()

    win = "horizon_panel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse)

    def set_step(x):
        global step
        step = 10 ** (x / 50.0 - 3)  # 0.001 – 1.0

    cv2.createTrackbar("step log", win, 50, 100, set_step)

    while cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1:
        show = undistort_and_draw()
        cv2.imshow(win, show)
        cv2.setWindowTitle(win,
                           f"horizon_panel | step={step:.3f} | "
                           f"k1={dist[0]:.4f} k2={dist[1]:.4f} k3={dist[4]:.4f} | "
                           f"p1={dist[2]:.4f} p2={dist[3]:.4f} | "
                           f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} | "
                           f"err={horizon_error(cv2.undistortPoints(np.array(pts_horizon, dtype=np.float32).reshape(-1, 1, 2), K, dist, P=new_K).reshape(-1, 2)) if len(pts_horizon) > 1 else 0:.2f}px"
                           )
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k == ord(' '):
            pts_horizon.clear()
        elif k == ord('z'):
            dist[:] = 0
            K[0, 0] = K[1, 1] = max(img_orig.shape[1], img_orig.shape[0]) * 0.8
            push_state()
            build_remap()
        elif k == ord('n'):
            if pop_state():
                build_remap()
        elif k == ord('c'):
            auto_calc()
        elif k == ord('x'):
            save_result(args.out, [img_orig.shape[1], img_orig.shape[0]])

        # 畸变调节
        elif k == ord('q'):
            dist[0] += step
            push_state()
            build_remap()
        elif k == ord('a'):
            dist[0] -= step
            push_state()
            build_remap()
        elif k == ord('w'):
            dist[1] += step
            push_state()
            build_remap()
        elif k == ord('s'):
            dist[1] -= step
            push_state()
            build_remap()
        elif k == ord('e'):
            dist[4] += step
            push_state()
            build_remap()
        elif k == ord('d'):
            dist[4] -= step
            push_state()
            build_remap()
        elif k == ord('r'):
            dist[2] += step
            push_state()
            build_remap()
        elif k == ord('f'):
            dist[2] -= step
            push_state()
            build_remap()
        elif k == ord('t'):
            dist[3] += step
            push_state()
            build_remap()
        elif k == ord('g'):
            dist[3] -= step
            push_state()
            build_remap()

        # 焦距调节
        elif k == ord('y'):
            K[0, 0] += step * 50
            K[1, 1] += step * 50
            push_state()
            build_remap()
        elif k == ord('h'):
            K[0, 0] -= step * 50
            K[1, 1] -= step * 50
            if K[0, 0] < 50:
                K[0, 0] = K[1, 1] = 50
            push_state()
            build_remap()
        elif k == ord('Y') - 32:  # Shift+y -> 单调 fx
            K[0, 0] += step * 50
            push_state()
            build_remap()
        elif k == ord('H') - 32:  # Shift+h -> 单调 fx
            K[0, 0] -= step * 50
            if K[0, 0] < 50:
                K[0, 0] = 50
            push_state()
            build_remap()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
