#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
← → 选帧  z 复位  n 撤销  x 保存  ESC 退出
滑块调节步长，实时生效
python tune_H_slider.py --input1 cam1.mp4 --input2 cam2.mp4
"""
import cv2
import numpy as np
import json
import argparse
import logging
import sys
import os
from pathlib import Path
from collections import deque

# ---------- gRPC ----------
sys.path.insert(0, os.path.abspath(r"D:\program\li3D-ML"))
import grpc
from grpc_detection import msg_pb2, msg_pb2_grpc

_HOST = '192.168.101.124'
_PORT = '7910'
conn = grpc.insecure_channel(_HOST + ':' + _PORT,
                             options=(('grpc.enable_http_proxy', 0),
                                      ('grpc.max_send_message_length', 10 * 4000 * 3000)))
client = msg_pb2_grpc.FormatDataStub(channel=conn)


def run(img, mode=3):
    img_shape = list(img.shape)
    img_bytes = bytes(img)
    return client.get_roi_from_bytes(
        msg_pb2.request_bytes(mode=str(mode), image=img_bytes, image_shape=img_shape, confidence=0)
    )


# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# ---------- 全局 ----------
H = np.eye(3, dtype=np.float64)
H_HISTORY = deque(maxlen=200)
H_INIT = None          # ★ 初始矩阵，复位用
STEP = {
    'scale': 0.001,
    'shear': 0.001,
    'trans': 1.0,
    'persp': 1e-5
}
FRAME_IDX = 0
CAP1, CAP2 = None, None


# ---------- 工具 ----------
def align_height(im1, im2, target_h=None):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    if target_h is None:
        target_h = max(h1, h2)
    scale1 = target_h / h1
    scale2 = target_h / h2
    im1 = cv2.resize(im1, (int(w1 * scale1), target_h), interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, (int(w2 * scale2), target_h), interpolation=cv2.INTER_AREA)
    return im1, im2


def load_intrinsic(json_path):
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    K = np.array(cfg['K'], dtype=np.float64)
    D = np.array(cfg['dist'], dtype=np.float64)
    h = int(cfg['img_size'][1])
    w = int(cfg['img_size'][0])
    return K, D, h, w


def img_undistort(img_bgr, K=None, D=None, cfg_height=None, cfg_width=None, alpha=0):
    if K is not None and D is not None:
        h, w = img_bgr.shape[:2]
        if cfg_height != h or cfg_width != w:
            raise ValueError(f"尺寸不匹配：图({h},{w})≠cfg({cfg_height},{cfg_width})")
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha)
        img_bgr = cv2.undistort(img_bgr, K, D, None, new_K)
    return img_bgr


# ---------- 滑块 ----------
def update_step(val, step_type):
    global STEP
    if step_type == 'scale':
        STEP['scale'] = max(0.0001, val / 10000.0)
    elif step_type == 'shear':
        STEP['shear'] = max(0.0001, val / 10000.0)
    elif step_type == 'trans':
        STEP['trans'] = max(0.1, val / 100.0)
    elif step_type == 'persp':
        STEP['persp'] = max(1e-6, val / 1000000.0)


# ---------- 键盘微调 ----------
def tune_matrix(H):
    global H_INIT
    k = cv2.waitKey(10) & 0xFF

    # 1. 一键复位
    if k == ord('z'):
        if H_INIT is not None:
            H[:] = H_INIT
            H_HISTORY.clear()
            H_HISTORY.append(H.copy())
            logging.info('复位到初始 H 矩阵')
        return k

    # 2. 撤销
    if k == ord('n'):
        if len(H_HISTORY) > 1:
            H_HISTORY.pop()
            H[:] = H_HISTORY[-1]
            logging.info('撤销一次矩阵修改')
        return k

    # 3. 真正修改才压栈
    modify_keys = {'q', 'a', 'w', 's', 'e', 'd', 'r', 'f', 't', 'g', 'y', 'h', 'u', 'j', 'i', 'k'}
    if chr(k) in modify_keys:
        H_HISTORY.append(H.copy())
        if k == ord('q'): H[0, 0] += STEP['scale']
        elif k == ord('a'): H[0, 0] -= STEP['scale']
        elif k == ord('w'): H[0, 1] += STEP['shear']
        elif k == ord('s'): H[0, 1] -= STEP['shear']
        elif k == ord('e'): H[0, 2] += STEP['trans']
        elif k == ord('d'): H[0, 2] -= STEP['trans']
        elif k == ord('r'): H[1, 0] += STEP['shear']
        elif k == ord('f'): H[1, 0] -= STEP['shear']
        elif k == ord('t'): H[1, 1] += STEP['scale']
        elif k == ord('g'): H[1, 1] -= STEP['scale']
        elif k == ord('y'): H[1, 2] += STEP['trans']
        elif k == ord('h'): H[1, 2] -= STEP['trans']
        elif k == ord('u'): H[2, 0] += STEP['persp']
        elif k == ord('j'): H[2, 0] -= STEP['persp']
        elif k == ord('i'): H[2, 1] += STEP['persp']
        elif k == ord('k'): H[2, 1] -= STEP['persp']
    return k


# ---------- 可视化 ----------
def make_vis(img1, img2, H, boxs, vis_h):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    warp = cv2.warpPerspective(img1, H, (w2, h2))
    alpha_1 = cv2.getTrackbarPos('Overlay Alpha_1', 'TUNE_H') / 100.0
    alpha_2 = cv2.getTrackbarPos('Overlay Alpha_2', 'TUNE_H') / 100.0
    over = cv2.addWeighted(img2, alpha_1, warp, alpha_2, 0)
    for (x1, y1, x2, y2) in boxs:
        src = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)[0]
        pts = np.int32(dst)
        cv2.polylines(over, [pts], True, (0, 255, 125), 2)
    if vis_h > 0:
        scale = vis_h / h2
        over = cv2.resize(over, (int(w2 * scale), vis_h), interpolation=cv2.INTER_AREA)
    return over


# ---------- 主流程 ----------
def main():
    global FRAME_IDX, CAP1, CAP2, H, H_HISTORY, H_INIT
    parser = argparse.ArgumentParser(description='← → 选帧  z 复位  n 撤销  x 保存  ESC 退出')
    parser.add_argument('--input1', type=str, default=r"./video/suqian/detail-night/cam1.mp4")
    parser.add_argument('--input2', type=str, default=r"./video/suqian/detail-night/cam2.mp4")
    parser.add_argument('--intrinsic1_dir', type=str,
                        default=r"D:\program\SuperGluePretrainedNetwork\camera_intrinsic\data\suqian_thermal\intrinsic.json")
    parser.add_argument('--intrinsic2_dir', type=str, default=None)
    parser.add_argument('--out', default='./json_file/suqian/夜间-细节/final_stable_homography.json')
    parser.add_argument('--initial_H', type=str,
                        default=r'./json_file/suqian/夜间-细节/final_stable_homography.json')
    parser.add_argument('--vis_h', type=int, default=480)
    parser.add_argument('--browse_h', type=int, default=480)
    args = parser.parse_args()

    # 去畸变
    K1 = D1 = h1 = w1 = None
    K2 = D2 = h2 = w2 = None
    if args.intrinsic1_dir and Path(args.intrinsic1_dir).exists():
        K1, D1, h1, w1 = load_intrinsic(args.intrinsic1_dir)
    if args.intrinsic2_dir and Path(args.intrinsic2_dir).exists():
        K2, D2, h2, w2 = load_intrinsic(args.intrinsic2_dir)

    CAP1 = cv2.VideoCapture(args.input1)
    CAP2 = cv2.VideoCapture(args.input2)
    if not CAP1.isOpened() or not CAP2.isOpened():
        logging.error('无法打开视频')
        return

    cv2.namedWindow('BROWSE', cv2.WINDOW_NORMAL)
    cv2.namedWindow('TUNE_H', cv2.WINDOW_NORMAL)

    # 加载初始 H
    if args.initial_H and Path(args.initial_H).exists():
        with open(args.initial_H, 'r') as f:
            initial_H = json.load(f)
        H = np.array(initial_H['H'], dtype=np.float64)
        logging.info(f'加载初始 H 矩阵：{args.initial_H}')
    else:
        H = np.eye(3)
    H_HISTORY.append(H.copy())
    H_INIT = H.copy()          # ★ 关键：浏览阶段就能复位

    # 创建滑块
    cv2.createTrackbar('Scale Step', 'TUNE_H', 100, 10000, lambda val: update_step(val, 'scale'))
    cv2.createTrackbar('Shear Step', 'TUNE_H', 100, 10000, lambda val: update_step(val, 'shear'))
    cv2.createTrackbar('Trans Step', 'TUNE_H', 100, 10000, lambda val: update_step(val, 'trans'))
    cv2.createTrackbar('Persp Step', 'TUNE_H', 100, 10000, lambda val: update_step(val, 'persp'))
    cv2.createTrackbar('Overlay Alpha_1', 'TUNE_H', 50, 100, lambda val: None)
    cv2.createTrackbar('Overlay Alpha_2', 'TUNE_H', 50, 100, lambda val: None)

    while True:
        CAP1.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
        CAP2.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
        ret1, fr1 = CAP1.read()
        ret2, fr2 = CAP2.read()
        if not (ret1 and ret2):
            logging.warning('到达视频末尾')
            break
        fr1_rgb = fr1.copy()
        fr2_rgb = fr2.copy()

        # 去畸变
        if K1 is not None and D1 is not None:
            fr1_rgb = img_undistort(fr1_rgb, K1, D1, h1, w1)
        if K2 is not None and D2 is not None:
            fr2_rgb = img_undistort(fr2_rgb, K2, D2, h2, w2)

        # 浏览窗口
        browse1, browse2 = align_height(fr1_rgb, fr2_rgb, args.browse_h)
        vis_browse = np.hstack([browse1, browse2])
        cv2.putText(vis_browse, f'Frame {FRAME_IDX}   <-  -> browse   Enter - tune   ESC - quit',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow('BROWSE', vis_browse)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 13:  # Enter -> 微调模式
            logging.info('进入原图微调模式（x 保存，z 复位，n 撤销，ESC 放弃）')
            H_HISTORY.clear()
            H_HISTORY.append(H.copy())
            while True:
                resp = run(fr1_rgb, mode=3)
                boxs = [(int(r.ROI[0]), int(r.ROI[1]), int(r.ROI[2]), int(r.ROI[3])) for r in resp.data]
                vis_tune = make_vis(fr1_rgb, fr2_rgb, H, boxs, args.vis_h)
                step_info = f"Step: scale={STEP['scale']:.4f} trans={STEP['trans']:.1f} persp={STEP['persp']:.6f}"
                cv2.putText(vis_tune, step_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('TUNE_H', vis_tune)

                k = tune_matrix(H)
                if k == ord('x'):  # 保存
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    json.dump({'H': H.tolist()}, Path(args.out).open('w'), indent=2)
                    logging.info(f'已保存 H → {args.out}')
                    return
                elif k == 27:  # ESC 放弃微调
                    logging.info('放弃微调')
                    break
        elif key == ord(','):  # 上一帧
            FRAME_IDX = max(0, FRAME_IDX - 1)
        elif key == ord('.'):  # 下一帧
            FRAME_IDX += 1

    cv2.destroyAllWindows()
    CAP1.release()
    CAP2.release()
    logging.info('未保存，直接退出')


if __name__ == '__main__':
    main()