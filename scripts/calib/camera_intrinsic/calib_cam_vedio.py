#!/usr/bin/env python3
"""
单目相机标定脚本（带可视化 & 实时输出）
用法:
    python calib_cam.py [--video ./data/video.mp4] [--w 9] [--h 6] [--square_size 28] [--fps taipu_main_side]
"""
import cv2, numpy as np, json, os, shutil, argparse
from pathlib import Path

def parse_args():
    data_path = r"../../../data/calib/camera_intrinsic/data"
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=rf"{data_path}/2025-08-13/192.168.101.116_01_2025081315033482.mp4", type=str)
    parser.add_argument('--w', default=12, type=int)
    parser.add_argument('--h', default=10, type=int)
    parser.add_argument('--square_size', default=52, type=float)
    parser.add_argument('--fps', default=1, type=float, help="处理的帧率（每秒处理的帧数）")
    return parser.parse_args()

def collect_frames(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的原始帧率
    frame_interval = int(fps / target_fps)  # 计算需要跳过的帧数
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:  # 每隔 frame_interval 帧保存一次
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def select_and_show(frames, pattern_size):
    selected, objpoints, imgpoints = [], [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    total, valid = 0, 0
    for img in frames:
        total += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 去噪
        gray = cv2.equalizeHist(gray)  # 增强对比度
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            valid += 1
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(img, pattern_size, corners, True)
            selected.append(img)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            cv2.putText(img, "No corners found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 可视化
        cv2.imshow("Calibration", img)
        key = cv2.waitKey(1)  # 展示 taipu_main_side 毫秒
        if key == 27:  # ESC 提前退出
            break
    cv2.destroyAllWindows()
    print(f"\n📊 总计 {total} 帧 | 有效 {valid} 帧 | 选中 {len(selected)} 帧")
    img_size = gray.shape[::-1] if frames else None
    return selected, objpoints, imgpoints, img_size

def calibrate(selected, objpoints, imgpoints, img_size, square_size):
    new_obj = [o * square_size for o in objpoints]
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        new_obj, imgpoints, img_size, None, None)
    print(f"📏 重投影误差(RMS): {ret:.4f} 像素")
    return K, D, ret

def save_selected(selected, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for idx, img in enumerate(selected):
        cv2.imwrite(str(Path(out_dir) / f"selected_{idx:04d}.png"), img)

def undistort(selected, K, D, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for idx, img in enumerate(selected):
        und = cv2.undistort(img, K, D)
        cv2.imwrite(str(Path(out_dir) / f"undistorted_{idx:04d}.png"), und)

def save_intrinsic(K, D, img_size, file='intrinsic.json'):
    data = {"K": K.tolist(),
            "dist": D.tolist(),
            "img_size": [img_size[0], img_size[1]]}
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 内参已保存到 {file}")

def main():
    args = parse_args()
    frames = collect_frames(args.video, args.fps)
    if not frames:
        print("⚠️ 未找到任何视频帧")
        return
    selected, objp, imgp, img_size = select_and_show(frames, (args.w, args.h))
    if len(selected) < 5:
        print("❌ 有效帧不足 5 帧，标定终止")
        return
    K, D, err = calibrate(selected, objp, imgp, img_size, args.square_size)
    video_dir = Path(args.video).parent
    save_selected(selected, f'{video_dir}/selected')
    undistort(selected, K, D, f'{video_dir}/undistorted')
    save_intrinsic(K, D, img_size, f'{video_dir}/intrinsic.json')

if __name__ == '__main__':
    main()