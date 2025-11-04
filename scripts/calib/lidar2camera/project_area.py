#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
draw_json_pixels.py
读取 saved_points.json（像素坐标）并在图上画点连线
用法：
    python draw_json_pixels.py --img <图片路径> --json <json路径> [--out <输出路径>]
"""
import cv2
import numpy as np
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="把 JSON 像素坐标画在图上并连线")
    parser.add_argument("--img",  default=r"D:\program\li3D-ML\data\JaXing\D\04_12_04_50_6a5fe126-a722-4f78-bff0-037402311c03_7\detail\det_img\det_trac_2025_11_04_12_04_55_863299_img.jpg")
    parser.add_argument("--json", default=r"D:\program\li3D-ML\data\calib\lidar2camera\data\jiaxing_capture\saved_points.json")
    parser.add_argument("--out",  default="./out.png", help="结果保存路径（默认在原图同名目录加 _lines）")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 读图
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)
    vis = img.copy()
    # vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2. 读像素坐标
    if not os.path.isfile(args.json):
        raise FileNotFoundError(args.json)
    with open(args.json, "r") as f:
        pts = json.load(f)          # [[x1,y1], [x2,y2], ...]

    pts = np.array(pts, dtype=int)  # 转 ndarray
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("JSON 格式应为 [[x1,y1], [x2,y2], ...]")

    # 3. 画点
    for x, y in pts:
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)  # 红点

    # 4. 画线（首尾相连）
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 5. 保存
    if args.out is None:
        base, ext = os.path.splitext(args.img)
        args.out = f"{base}_lines{ext}"
    cv2.imwrite(args.out, vis)
    print(f"已保存到 {args.out}")

    # 6. 可选：弹窗看结果
    cv2.imshow("lines", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()