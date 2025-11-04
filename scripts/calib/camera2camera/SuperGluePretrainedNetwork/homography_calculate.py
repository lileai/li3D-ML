import time
from pathlib import Path
import sys
import os
import glob
import argparse
import cv2
import matplotlib.cm as cm
import json
import logging
import torch
import numpy as np
from collections import deque

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

sys.path.insert(0, os.path.abspath(r"D:\program\li3D-ML"))

import grpc
from grpc_detection import msg_pb2, msg_pb2_grpc

_HOST = '192.168.101.124'  # 设置grpc服务ip
_PORT = '7910'  # 设置grpc服务端口
CURRENT_REGION = 0

conn = grpc.insecure_channel(_HOST + ':' + _PORT, options=(('grpc.enable_http_proxy', 0),
                                                           ('grpc.max_send_message_length', 10 * 4000 * 3000)))  # 监听频道
client = msg_pb2_grpc.FormatDataStub(channel=conn)  # 客户端使用Stub类发送请求,参数为频道,为了绑定链接


def run(img, mode=3):
    '''
    客户端练级服务端测试用的脚本
    :return: 无返回，但是会print服务端的输出
    '''

    img_shape = list(img.shape)
    img = bytes(img)
    response = client.get_roi_from_bytes(
        msg_pb2.request_bytes(
            mode=str(mode), image=img, image_shape=img_shape, confidence=0
        )
    )

    return response


torch.set_grad_enabled(False)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------- 日志 ----------
def setup_logger(level='INFO'):
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[logging.StreamHandler()]
    )


# ---------- 滑窗 RANSAC（平面 inliers） ----------
def stable_planar_from_queue(q, ransac_th=5.0):
    if len(q) == 0:
        return None, None
    all1 = np.vstack([p[0] for p in q])
    all2 = np.vstack([p[1] for p in q])
    H, mask = cv2.findHomography(all1, all2, cv2.USAC_PROSAC, ransac_th)
    if H is None:
        return None, None
    inliers1 = all1[mask.ravel().astype(bool)]
    inliers2 = all2[mask.ravel().astype(bool)]
    return H, (inliers1, inliers2)


def color_box_only_blend(rgb1, rgb2, H_list, boxs, area=None, vis_h=360,
                         color=(0, 125, 255), thickness=2,
                         boundaries=None):
    """
    boundaries : 写死分区右边界像素列表，长度 = 区域数
                例 [600, 1200, 1920] 表示
                  区域1 [0,600)
                  区域2 [600,1200)
                  区域3 [1200,1920]
                默认 None → 等分
    """

    def maybe_resize(img):
        if vis_h <= 0:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * vis_h / h), vis_h),
                          interpolation=cv2.INTER_AREA)

    # 1. 兼容单个 H
    if not isinstance(H_list, list):
        H_list = [H_list]
    region_num = len(H_list)
    assert region_num > 0, "H_list 不能为空"

    img_w = rgb1.shape[1]

    # 2. 生成 reg_id 计算器
    if boundaries is None:  # 等分
        def get_reg_id(cx):
            return int(cx / (img_w / region_num))
    else:  # 写死
        boundaries = np.asarray(boundaries, dtype=int)
        assert len(boundaries) == region_num and boundaries[-1] == img_w, \
            "boundaries 长度必须等于区域数，且最后一项等于图像宽"

        def get_reg_id(cx):
            return int(np.searchsorted(boundaries, cx, side='right'))

    h2, w2 = rgb2.shape[:2]

    # 3. 若指定 area，提前过滤
    if area is not None:
        target = int(area)
        boxs = [(x1, y1, x2, y2) for x1, y1, x2, y2 in boxs
                if get_reg_id((x1 + x2) * 0.5) + 1 == target]
        if not boxs:
            # 先各自 resize 到 vis_h，再拼，宽度锁定为 rgb1.w + rgb2.w
            r1 = maybe_resize(rgb1)
            r2 = maybe_resize(rgb2)
            empty = np.hstack([r1, r2])
            return empty

    # 4. 投影（统一用 get_reg_id）
    dst_pts = []
    for x1, y1, x2, y2 in boxs:
        reg_id = get_reg_id((x1 + x2) * 0.5)
        reg_id = np.clip(reg_id, 0, region_num - 1)
        H = H_list[reg_id]
        src = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)[0]
        dst_pts.append(dst)
    dst_pts = np.array(dst_pts)

    # 5. 画框、写字、缩放、拼接（完全不变）
    rgb1_with_box = rgb1.copy()
    for x1, y1, x2, y2 in boxs:
        cv2.rectangle(rgb1_with_box, (int(x1), int(y1)),
                      (int(x2), int(y2)), color, thickness)

    rgb2_with_box = rgb2.copy()
    text_thick = 1
    for i, pts in enumerate(dst_pts):
        x1, _, x2, _ = boxs[i]
        reg_id = get_reg_id((x1 + x2) * 0.5)
        reg_id = np.clip(reg_id, 0, region_num - 1)
        label = f"{reg_id + 1}"

        pts_int = np.int32(pts)
        # 计算外接矩形框
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # 画框
        cv2.rectangle(rgb2_with_box, (x_min, y_min), (x_max, y_max), color, thickness)
        # cv2.polylines(rgb2_with_box, [pts_int], True, color, thickness)
        box_h = int(pts_int[:, 1].max() - pts_int[:, 1].min())
        font_scale = np.clip(box_h / 60.0, 0.3, 1.0)
        xs, ys = pts_int[:, 0], pts_int[:, 1]
        tx = int((xs.min() + xs.max()) / 2)
        ty = int(ys.min())
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, text_thick)
        org = (tx - tw // 2, ty - 4)
        cv2.rectangle(rgb2_with_box,
                      (org[0] - 2, org[1] - th - 2),
                      (org[0] + tw + 2, org[1] + 2),
                      (0, 0, 0), -1)
        cv2.putText(rgb2_with_box, label, org,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 255, 0), text_thick, cv2.LINE_AA)

    return np.hstack([maybe_resize(rgb1_with_box),
                      maybe_resize(rgb2_with_box)])


def overlay(img1, img2, alpha=0.5):
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def make_vis_dual_mode(rgb1, rgb2, H_list, boxs, area=None, vis_h=360,
                       color=(0, 125, 255), thickness=2, boundaries=None, mode='calib'):
    """
    mode == 'calib' : 在中间插入一张 rgb1->rgb2 的透视半透明图
    mode == 'show'  : 保持原来 color_box_only_blend 逻辑
    """
    # 1. 先拿原来的三拼图（带框）
    base_vis = color_box_only_blend(rgb1, rgb2, H_list, boxs, area,
                                    vis_h, color, thickness, boundaries)
    if mode == 'show':  # 无改动
        return base_vis

    # 2. calib 模式：把中间换成 warpOverlay
    h, w_tot = base_vis.shape[:2]
    w_single = w_tot // 3  # 每拼宽度

    # 2.1 生成 rgb1->rgb2 透视图
    if H_list is None:
        warp = rgb1  # 无 H 时退化为原图
    else:
        # 兼容多 H：用中间区域对应的 H 做整幅 warp
        if isinstance(H_list, list):
            H = H_list[len(H_list) // 2]
        else:
            H = H_list
        warp = cv2.warpPerspective(rgb1, H, (rgb2.shape[1], rgb2.shape[0]))

    # 2.2 透明叠加
    over = overlay(rgb2, warp, 0.5)

    # 2.3 缩放成统一高度
    def resize(img):
        if vis_h <= 0:
            return img
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * vis_h / h), vis_h))

    left = resize(base_vis[:, :w_single, :])
    right = resize(base_vis[:, -w_single:, :])
    middle = resize(over)

    return np.hstack([left, middle, right])


def load_intrinsic(json_path):
    """返回 K(3×3), D(1×5), h, w"""
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    K = np.array(cfg['K'], dtype=np.float64)
    D = np.array(cfg['dist'][0], dtype=np.float64)   # 5 系数
    h = int(cfg['img_size'][1])        # 高
    w = int(cfg['img_size'][0])        # 宽
    return K, D, h, w


def load_stable_H(path):
    """
    根据路径类型加载 H 矩阵：
    - 单个 JSON 文件：返回 H（3×3 numpy 数组）
    - 文件夹：返回 [H1, H2, ...]（list of 3×3 arrays）
    """
    if os.path.isfile(path) and path.lower().endswith('.json'):
        # 单个文件：直接返回 H
        with open(path, 'r') as f:
            cfg = json.load(f)
        return np.array(cfg['H'], dtype=np.float64)

    elif os.path.isdir(path):
        # 文件夹：读取所有 json 文件，返回 H 列表
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


# ---------- 保存最终 H ----------
def save_final_H(H, frame_id, win_len, path=None, ransac_th=5.0):
    out = {
        "H": H.tolist(),
        "frame_id": frame_id,
        "win_len": win_len,
        "ransac_th": ransac_th
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    logging.info(f'最终稳定 H 已写入 {path}')


def process_resize(w, h, resize):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def recover_original_homography(H_resize,
                                w1, h1,  # 图 1 原始宽高
                                w2, h2,  # 图 2 原始宽高
                                w_new, h_new):  # 统一后的宽高
    """
    把在 (w_new × h_new) 图上算出的 H_resize
    升尺度回原始分辨率下的 H_original
    返回 3×3 ndarray
    """
    # 图 1：原始 → resize
    S1 = np.array([[w_new / w1, 0, 0],
                   [0, h_new / h1, 0],
                   [0, 0, 1]], dtype=float)

    # 图 2：原始 → resize
    S2 = np.array([[w_new / w2, 0, 0],
                   [0, h_new / h2, 0],
                   [0, 0, 1]], dtype=float)

    # 逆映射：resize → 原始
    H_original = np.linalg.inv(S2) @ H_resize @ S1
    return H_original


# -------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='整图彩色 + 平面 RANSAC + Torch-LM 三栏')
    # 新增模式选择
    parser.add_argument('--mode', choices=['calib', 'show'], default='show',
                        help='calib: 三拼[rgb1|overlay|rgb2]并在线计算H; '
                             'show: 双拼[rgb1|rgb2]直接加载H')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--input1', type=str, default=r"D:\program\li3D-ML\data\Qiaolin\thermal\thermal\1_to_2.mp4")
    parser.add_argument('--input2', type=str, default=r"D:\program\li3D-ML\data\Qiaolin\thermal\transform_1_to_2\1_to_2.mp4")
    parser.add_argument('--intrinsic1_dir', type=str,
                        default=r"D:\program\li3D-ML\scripts\calib\camera_intrinsic\data\qiaolin_thermal\intrinsic.json")
    parser.add_argument('--intrinsic2_dir', type=str,
                        default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_video', type=str, default=r"./video/1.avi")
    parser.add_argument('--best_h_json', type=str, default=None, help="支持输入矩阵的文件夹或者单独的文件路径")
    parser.add_argument('--skip', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=1000000)
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480])
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor')
    parser.add_argument('--max_keypoints', type=int, default=-1)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--nms_radius', type=int, default=1, help="标定的时候需要改成1，默认为10")
    parser.add_argument('--sinkhorn_iterations', type=int, default=20)
    parser.add_argument('--match_threshold', type=float, default=0.2)
    parser.add_argument('--ransac_th', type=float, default=5.0)
    parser.add_argument('--no_display', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--win_len', default=100, help="动态标定窗口大小为25~50，静态标定窗口大小为100~200")
    parser.add_argument('--vis_h', type=int, default=480, help='可视化高度，-1=原图')
    parser.add_argument('--net', type=int, default=True, help="标定时为True")
    parser.add_argument('--boundaries', type=str, default=None, help="自定义区域划分，标定时为None")
    parser.add_argument('--waitkey', type=str, default=0, help="标定时为0")

    # 先解析 mode，再覆盖默认值
    args, _ = parser.parse_known_args()
    if args.mode == 'calib':
        parser.set_defaults(net=True, skip=1, nms_radius=10, win_len=100, boundaries=None, waitkey=0,
                            output_video='./video/calib.avi',
                            best_h_json='./json_file/thermal_distort/final_stable_homography_2.json'),
    else:
        # 桥林云台+星光枪分区[500, 1280]
        # 热成像分区[190, 384]
        parser.set_defaults(net=False, nms_radius=10, win_len=100, boundaries=[190, 384], waitkey=1,
                            output_video='./video/show.avi',
                            best_h_json='./json_file/thermal_distort')
    opt = parser.parse_args()

    setup_logger(opt.log_level)
    logging.info('Starting Video Matching with Logging')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    # 去畸变
    if opt.intrinsic1_dir is not None:
        K1, D1, h1, w1 = load_intrinsic(opt.intrinsic1_dir)
    else:
        K1 = D1 = h1 = w1 = None

    if opt.intrinsic2_dir is not None:
        K2, D2, h2, w2 = load_intrinsic(opt.intrinsic2_dir)
    else:
        K2 = D2 = h2 = w2 = None

    vs1 = VideoStreamer(opt.input1, skip=opt.skip,
                        image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=opt.max_length, K=K1, D=D1, h=h1, w=w1)
    vs2 = VideoStreamer(opt.input2, skip=opt.skip,
                        image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=opt.max_length, K=K2, D=D2, h=h2, w=w2)

    if not opt.no_display:
        cv2.namedWindow('Full Color', cv2.WINDOW_NORMAL)

    match_queue = deque(maxlen=6000)
    out_video = None
    timer = AverageTimer()
    _last_H = None  # 用于检测 H 是否变化
    frame_num = 0
    while True:
        frame_num += 1
        area = 1
        if frame_num >= 518:
            area = 2
        frame1, frame1_rgb, ret1 = vs1.next_frame()
        frame2, frame2_rgb, ret2 = vs2.next_frame()
        w_new1, h_new1 = process_resize(frame1.shape[1], frame1.shape[0], opt.resize)
        w_new2, h_new2 = process_resize(frame2.shape[1], frame2.shape[0], opt.resize)

        frame1 = cv2.resize(frame1, (w_new1, h_new1), interpolation=cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, (w_new2, h_new2), interpolation=cv2.INTER_AREA)

        logging.info(f"当前是第{frame_num}帧")
        # 调用grpc检测
        response = run(frame1_rgb)
        boxs = []
        for i in range(len(response.data)):
            boxs.append(response.data[i].ROI)
        if not ret1 or not ret2:
            logging.info('Finished processing all frames')
            break
        timer.update('data')
        if opt.net:
            # 灰度 tensor 给网络
            f1_tensor = frame2tensor(frame1, device)
            f2_tensor = frame2tensor(frame2, device)
            pred = matching({'image0': f1_tensor, 'image1': f2_tensor})

            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            valid = matches > -1

            if valid.sum() < 4:
                H_stable = None
            else:
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                # 只把 RANSAC 平面 inliers 入队
                H_ransac, inlier_pack = stable_planar_from_queue([(mkpts0, mkpts1)])
                if H_ransac is not None and inlier_pack is not None:
                    in0, in1 = inlier_pack
                    match_queue.append((in0, in1))
                    yes = True
                    if len(match_queue) >= min(opt.win_len, 200) and yes:
                        H_stable, _ = stable_planar_from_queue(match_queue)
                    else:
                        H_stable = H_ransac
                    H_stable = recover_original_homography(
                        H_stable,
                        frame1_rgb.shape[1], frame1_rgb.shape[0],
                        frame2_rgb.shape[1], frame2_rgb.shape[0],
                        w_new1, h_new1
                    )
                else:
                    H_stable = None
            if H_stable is not None:
                # 第一次 || 与上一次不同 → 才保存
                if (_last_H is None) or (not np.allclose(H_stable, _last_H, atol=1e-4, rtol=1e-4)):
                    save_final_H(H_stable, vs1.i - 1, len(match_queue), opt.best_h_json, opt.ransac_th)
                    _last_H = H_stable.copy()  # 更新缓存
                    logging.info('最终稳定 H 已更新（矩阵发生变化）')
                else:
                    logging.debug('最终稳定 H 无变化，跳过写入')
            else:
                logging.warning('未获得有效 H，不保存 JSON')
        else:
            H_stable = load_stable_H(opt.best_h_json)
        # 用彩色原图做可视化
        if H_stable is not None:
            vis = make_vis_dual_mode(frame1_rgb, frame2_rgb, H_stable, boxs, area=area, vis_h=opt.vis_h,
                                     boundaries=opt.boundaries, mode=opt.mode)
        else:
            confidence = pred['matching_scores0'][0].cpu().numpy()
            color = cm.jet(confidence[valid])
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0))
            ]
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
            ]
            vis = make_matching_plot_fast(
                frame1, frame2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=opt.max_keypoints, small_text=small_text)

        if out_video is None and opt.output_video:
            h_vis, w_vis = vis.shape[:2]
            if vis is None or vis.size == 0:
                logging.error('可视化帧为空，无法初始化 VideoWriter')
                opt.output_video = None
            else:
                out_fps = vs1.fps / opt.skip
                logging.info(f'VideoWriter: fps={out_fps}, size={w_vis}x{h_vis}')
                os.makedirs(os.path.dirname(opt.output_video), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_video = cv2.VideoWriter(opt.output_video, fourcc,
                                            out_fps, (w_vis, h_vis))
                if not out_video.isOpened():
                    raise RuntimeError(f'VideoWriter 无法打开：{opt.output_video}')

        if not opt.no_display:
            cv2.imshow('Full Color', vis)
            if cv2.waitKey(opt.waitkey) & 0xFF in (27, ord('q')):
                break

        if opt.output_dir is not None:
            Path(opt.output_dir).mkdir(exist_ok=True)
            out_file = Path(opt.output_dir, f'color_{vs1.i - 1:06}_{vs2.i - 1:06}.png')
            cv2.imwrite(str(out_file), vis)

        if out_video is not None:
            if vis.shape[:2] != (h_vis, w_vis):
                logging.warning(f'帧尺寸突变: {vis.shape[:2]} ≠ {(h_vis, w_vis)}')
                vis = cv2.resize(vis, (w_vis, h_vis))
            out_video.write(vis)
        # if out_video is not None:
        #     out_video.write(vis)
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()
    vs1.cleanup()
    vs2.cleanup()
