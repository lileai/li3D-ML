#!/usr/bin/env python3
import cv2
import os
import argparse
from tqdm import tqdm

def video2images(video_path, out_dir, interval=1, resize=None, max_frames=None):
    """
    将视频按帧保存为图片
    :param video_path: 输入视频文件路径
    :param out_dir:    图片输出目录
    :param interval:   每间隔多少帧保存一张（1=每帧都存）
    :param resize:     输出尺寸 tuple(w, h)，None 表示不缩放
    :param max_frames: 最多保存多少张（None=全部）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(out_dir, exist_ok=True)

    digit = len(str(total_frames))          # 文件名零填充位数
    count_saved = 0
    count_read = 0

    pbar = tqdm(total=total_frames, desc="Extracting")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count_read += 1
        pbar.update(1)

        # 只保存指定间隔
        if count_read % interval != 0:
            continue

        # 缩放
        if resize:
            frame = cv2.resize(frame, resize)

        out_name = os.path.join(out_dir, f"frame_{count_saved:0{digit}d}.jpg")
        cv2.imwrite(out_name, frame)
        count_saved += 1

        if max_frames and count_saved >= max_frames:
            break

    cap.release()
    pbar.close()
    print(f"Done! {count_saved} images saved to -> {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video to images extractor")
    parser.add_argument("-video", default=r"D:\program\li3D-ML\data\JaXing\redis_video_20251030_133657.avi", help="path to input video")
    parser.add_argument("-o", "--out", default="./images", help="output folder (default: images)")
    parser.add_argument("-i", "--interval", type=int, default=10, help="save 1 image every N frames (default: 1)")
    parser.add_argument("-s", "--size", type=str, default=None,
                        help="output size WxH, e.g. 640x480 (default: original)")
    parser.add_argument("-m", "--max", type=int, default=None,
                        help="max number of images to save (default: all)")
    args = parser.parse_args()

    resize = None
    if args.size:
        w, h = map(int, args.size.split("x"))
        resize = (w, h)

    video2images(args.video, args.out, interval=args.interval, resize=resize, max_frames=args.max)