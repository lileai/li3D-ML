import os
import glob
import argparse
import cv2
import json
import logging
import torch
import numpy as np
from models.utils import VideoStreamer
import grpc
from grpc_detection import msg_pb2, msg_pb2_grpc
import time
from collections import deque
from functools import lru_cache

_HOST = '192.168.101.124'
_PORT = '7910'
CURRENT_REGION = 0
conn = grpc.insecure_channel(_HOST + ':' + _PORT,
                             options=(('grpc.enable_http_proxy', 0),
                                      ('grpc.max_send_message_length', 10 * 4000 * 3000)))
client = msg_pb2_grpc.FormatDataStub(channel=conn)

def run(img, mode=3):
    img_shape = list(img.shape)
    img = bytes(img)
    response = client.get_roi_from_bytes(
        msg_pb2.request_bytes(mode=str(mode), image=img, image_shape=img_shape, confidence=0))
    return response

torch.set_grad_enabled(False)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_logger(level='INFO'):
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[logging.StreamHandler()]
    )

def load_intrinsic(json_path):
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    K = np.array(cfg['K'], dtype=np.float64)
    D = np.array(cfg['dist'], dtype=np.float64)
    h = int(cfg['img_size'][1])
    w = int(cfg['img_size'][0])
    return K, D, h, w

def load_stable_H(path):
    if os.path.isfile(path) and path.lower().endswith('.json'):
        with open(path, 'r') as f:
            cfg = json.load(f)
        return [np.array(cfg['H'], dtype=np.float64)]
    elif os.path.isdir(path):
        json_files = sorted(glob.glob(os.path.join(path, "*.json")))
        if not json_files:
            raise ValueError(f"文件夹中未找到 JSON 文件：{path}")
        H_list = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                cfg = json.load(f)
            H_list.append(np.array(cfg['H'], dtype=np.float64))
        return H_list
    else:
        raise FileNotFoundError(f"路径无效（不是 .json 文件也不是文件夹）：{path}")


# ---------------- 速度窗口 ----------------
class PixelVelocityWindow:
    def __init__(self, delta=5, alpha=0.12):
        self.delta = delta          # 与 N 帧前比较
        self.alpha = alpha          # 平滑系数
        self.buf = deque(maxlen=delta + 1)
        self.vx_lpf = 0.0
        self.vy_lpf = 0.0

    def update(self, t, x, y):
        """返回平滑后的 像素/帧（vs delta 帧前）"""
        self.buf.append((t, x, y))
        if len(self.buf) < self.delta + 1:
            return None, None
        x0, y0 = self.buf[0][1], self.buf[0][2]
        x1, y1 = self.buf[-1][1], self.buf[-1][2]
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return None, None
        vx = (x1 - x0) / self.delta
        vy = (y1 - y0) / self.delta
        # 首次初始化
        if self.vx_lpf == 0.0 and self.vy_lpf == 0.0:
            self.vx_lpf = vx
            self.vy_lpf = vy
        else:
            self.vx_lpf = self.alpha * vx + (1 - self.alpha) * self.vx_lpf
            self.vy_lpf = self.alpha * vy + (1 - self.alpha) * self.vy_lpf
        return self.vx_lpf, self.vy_lpf

    # 原框顶画速度（无黑底）
    def draw_speed_orig(self, img, vx, vy, x1, y1, x2, y2, color=(0, 255, 0)):
        if vx is None:
            txt = "..."
        else:
            txt = f"{vx:+.1f} pix/f"
        box_h = int(y2 - y1)
        scale = max(0.3, box_h / 80.0)
        thick = max(1, int(box_h / 100))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (int(x1 + (x2 - x1 - tw) * 0.5), int(y1 - 8))
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    # 映射框顶画速度（无黑底）
    def draw_speed_mapped(self, img, vx, vy, pts, color=(0, 255, 0)):
        if vx is None:
            txt = "..."
        else:
            txt = f"{vx:+.1f} pix/f"
        box_h = int(pts[:, 1].max() - pts[:, 1].min())
        scale = max(0.3, box_h / 80.0)
        thick = max(1, int(box_h / 100))
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        xs, ys = pts[:, 0], pts[:, 1]
        cx = int((xs.min() + xs.max()) / 2)
        top = int(ys.min())
        org = (cx - tw // 2, top - 8)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ---------------- 通用区域号计算器 ----------------
@lru_cache(maxsize=1)
def _reg_id_func(img_w, region_num, boundaries_tuple):
    """返回：根据 cx 计算区域号的函数"""
    if boundaries_tuple is None:
        def get_reg_id(cx):
            return int(cx / (img_w / region_num))
    else:
        boundaries = np.asarray(boundaries_tuple, dtype=int)
        assert len(boundaries) == region_num and boundaries[-1] == img_w, \
            "boundaries 长度必须等于 H_list 长度，且最后一项等于图像宽"
        def get_reg_id(cx):
            return int(np.searchsorted(boundaries, cx, side='right'))
    return get_reg_id


# ---------------- 数值计算封装 ----------------
def compute_frame(t_now, boxs, H_list, boundaries, vw_orig, vw_map, img_w, area=None):
    """返回：完整字典 or None（无框或区域不匹配）"""
    if not boxs:
        return None

    # 1. 区域数 = H 数量（强制校验）
    region_num = len(H_list)
    if boundaries is not None:
        boundaries = np.asarray(boundaries, dtype=int)
        assert len(boundaries) == region_num and boundaries[-1] == img_w, \
            "boundaries 长度必须等于 H_list 长度，且最后一项等于图像宽"

    # 2. 区域号计算器（通用）
    get_reg_id = _reg_id_func(img_w, region_num, tuple(boundaries) if boundaries else None)

    # 3. 首框区域号
    x1, y1, x2, y2 = boxs[0]
    reg_id = get_reg_id((x1 + x2) * 0.5)
    reg_id = np.clip(reg_id, 0, region_num - 1)
    preset_id = reg_id + 1  # 从 1 开始

    # 4. area 过滤（与原来完全一致）
    if area is not None and preset_id != int(area):
        return None

    # 5. 原框速度（可能 None）
    vx_orig, vy_orig = vw_orig.update(t_now, x1, y1)
    # 6. 映射后 4 点 + 最小外接矩形
    H = H_list[reg_id] if isinstance(H_list, list) else H_list
    src = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    pts = cv2.perspectiveTransform(src, H)[0]
    xs, ys = pts[:, 0], pts[:, 1]
    mapped_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    # 7. 映射框速度（中心点）
    mx = (mapped_box[0] + mapped_box[2]) * 0.5
    my = (mapped_box[1] + mapped_box[3]) * 0.5
    vx_map, vy_map = vw_map.update(t_now, mx, my)

    return {
        'orig_box': [x1, y1, x2, y2],  # 原始框
        'orig_vx': vx_orig if vx_orig is not None else 0.0,  # 原始框x方向的速度
        'orig_vy': vy_orig if vy_orig is not None else 0.0,  # 原始框y方向的速度
        'mapped_pts': pts.tolist(),  # 映射框
        'mapped_box': mapped_box, # 映射框的最小外接矩形框
        'mapped_vx': vx_map if vx_map is not None else 0.0,  # 映射框x方向的速度
        'mapped_vy': vy_map if vy_map is not None else 0.0,  # 映射框y方向的速度
        'preset_id': preset_id  # 框在星光枪/热成像中的区域id（预置点）
    }

# ---------------- 可视化 ----------------
def draw_vis(rgb1, rgb2, H_list, boxs, vw_orig, vw_map,
             info, color=(0, 125, 255), thickness=2, boundaries=None, vis_h=360):
    def maybe_resize(img):
        if vis_h <= 0:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * vis_h / h), vis_h), interpolation=cv2.INTER_AREA)

    rgb1_out = rgb1.copy()
    rgb2_out = rgb2.copy()
    if info is None:  # 无框或区域不匹配
        return np.hstack([maybe_resize(rgb1_out), maybe_resize(rgb2_out)])

    # 1. 原框画框+速度
    x1, y1, x2, y2 = info['orig_box']
    thick = max(1, int((y2 - y1) / 100))
    cv2.rectangle(rgb1_out, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
    vw_orig.draw_speed_orig(rgb1_out, info['orig_vx'], info['orig_vy'], x1, y1, x2, y2)

    # 2. 映射框画框+速度
    pts_int = np.int32(info['mapped_pts'])
    x_min, y_min, x_max, y_max = info['mapped_box']
    thick = max(1, int((y_max - y_min) / 100))
    cv2.rectangle(rgb2_out, (x_min, y_min), (x_max, y_max), color, thick)
    # 预置点标签
    label = str(info['preset_id'])
    font_scale = max(0.3, (y_max - y_min) / 80.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    org = (int((x_min + x_max) / 2 - tw // 2), int(y_min - 4))
    cv2.rectangle(rgb2_out, (org[0] - 2, org[1] - th - 2), (org[0] + tw + 2, org[1] + 2), (0, 0, 0), -1)
    cv2.putText(rgb2_out, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thick, cv2.LINE_AA)
    vw_map.draw_speed_mapped(rgb2_out, info['mapped_vx'], info['mapped_vy'], pts_int)

    return np.hstack([maybe_resize(rgb1_out), maybe_resize(rgb2_out)])

# -------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='单应性推理')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--input1', type=str, default=r"./video/suqian/complete-night/cam1.mp4", help="支持相机流输入")
    parser.add_argument('--input2', type=str, default=r"./video/suqian/complete-night/cam2.mp4", help="支持相机流输入")
    parser.add_argument('--intrinsic1_dir', type=str, default=r"D:\program\SuperGluePretrainedNetwork\camera_intrinsic\data\suqian_thermal\intrinsic.json")
    parser.add_argument('--intrinsic2_dir', type=str, default=None)
    parser.add_argument('--best_h_json', type=str, default=r"./json_file/suqian/夜间-全貌/final_stable_homography.json",help="支持输入矩阵的文件夹或者单独的文件路径")
    parser.add_argument('--out_video', type=str, default=r"D:\program\SuperGluePretrainedNetwork\video\suqian\output\complete-night.mp4")

    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--resize', type=int, nargs='+', default=[-1])
    parser.add_argument('--max_length', type=int, default=1000000)
    parser.add_argument('--boundaries', type=str, default=None, help="自定义区域划分")
    parser.add_argument('--area', type=int, default=None, help="只处理指定区域号（从 1 开始）")
    opt = parser.parse_args()

    setup_logger(opt.log_level)
    logging.info('Starting Video Matching with Logging')

    if opt.intrinsic1_dir is not None:
        K1, D1, h1, w1 = load_intrinsic(opt.intrinsic1_dir)
    else:
        K1 = D1 = h1 = w1 = None

    if opt.intrinsic2_dir is not None:
        K2, D2, h2, w2 = load_intrinsic(opt.intrinsic2_dir)
    else:
        K2 = D2 = h2 = w2 = None

    H_stable = load_stable_H(opt.best_h_json)

    vs1 = VideoStreamer(opt.input1, skip=opt.skip,
                        image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=opt.max_length, K=K1, D=D1, h=h1, w=w1)
    vs2 = VideoStreamer(opt.input2, skip=opt.skip,
                        image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=opt.max_length, K=K2, D=D2, h=h2, w=w2)
    frame_num = 0

    vw_orig = PixelVelocityWindow(delta=5, alpha=0.12)
    vw_map  = PixelVelocityWindow(delta=5, alpha=0.12)
    out = None

    while True:
        frame_num += 1
        frame1, frame1_rgb, ret1 = vs1.next_frame()
        frame2, frame2_rgb, ret2 = vs2.next_frame()
        if not ret1:
            logging.info('Finished processing all frames')
            break
        response = run(frame1_rgb)
        boxs = [response.data[i].ROI for i in range(len(response.data))]

        # 1. 数值计算（封装字典）
        t_now = time.time()
        info = compute_frame(t_now, boxs, H_stable, opt.boundaries, vw_orig, vw_map,
                             frame1_rgb.shape[1], area=opt.area)

        # 2. 可视化（只读字典）
        vis = draw_vis(frame1_rgb, frame2_rgb,
                       H_stable,
                       boxs,
                       vw_orig, vw_map,
                       info,
                       boundaries=opt.boundaries, vis_h=520)

        # 3. 保存 & 显示
        if out is None:
            h, w = vis.shape[:2]
            out_fps = vs1.fps / opt.skip
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(opt.out_video, fourcc, out_fps, (w, h))
        out.write(vis)
        cv2.imshow('Full Color', vis)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    if out is not None:
        out.release()
    cv2.destroyAllWindows()