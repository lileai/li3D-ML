import os
import numpy as np
import cv2
import json

def load_intrinsic(json_path):
    """返回 K(3×3), D(1×5), h, w"""
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    K = np.array(cfg['K'], dtype=np.float64)
    D = np.array(cfg['dist'][0], dtype=np.float64)   # 5 系数
    h = int(cfg['img_size'][1])        # 高
    w = int(cfg['img_size'][0])        # 宽
    return K, D, h, w


def img_undistort(img_bgr, K=None, D=None, cfg_height=None, cfg_width=None, alpha=0.5):
    if K is not None and D is not None:
        # 1. 获取最优相机矩阵 + 有效 ROI
        h, w = img_bgr.shape[:2]
        if cfg_height != h or cfg_width != w:
            raise ValueError(f"尺寸不匹配：图({h},{w})≠cfg({cfg_height},{cfg_width})")
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, D, (w, h), alpha=alpha)  # alpha=0 保留最多像素

        # 2. 用新矩阵去畸变（不会裁掉有效区域）
        img_bgr = cv2.undistort(img_bgr, K, D, None, new_K)
    return img_bgr

if __name__ == '__main__':
    json_path = r'D:\program\SuperGluePretrainedNetwork\camera_intrinsic\data\qiaolin_thermal\intrinsic.json'
    img_path = r'D:\program\SuperGluePretrainedNetwork\camera_intrinsic\data\qiaolin_thermal\frame_0000.jpg'
    img_bgr = cv2.imread(img_path)
    K, D, h, w = load_intrinsic(json_path)
    img_bgr = img_undistort(img_bgr, K, D, h, w, alpha=1)
    save_path = os.path.splitext(img_path)[0] + '_undistort.png'
    cv2.imwrite(save_path, img_bgr)
    print('saved:', save_path)
