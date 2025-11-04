import cv2
import json
import numpy as np

# ---------- 1. 读内参 ----------
def load_intrinsic(json_path):
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    K = np.array(cfg['K'], dtype=np.float64)
    D = np.array(cfg['dist'], dtype=np.float64)
    h, w = int(cfg['img_size'][1]), int(cfg['img_size'][0])
    return K, D, h, w

# ---------- 2. 单点畸变矫正 ----------
def undistort_points(points, K, D, img_size=None, alpha=0):
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    h, w = img_size[::-1] if img_size else (int(K[1, 2]*2), int(K[0, 2]*2))
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha)
    pts_und = cv2.undistortPoints(points.reshape(-1, 1, 2), K, D, P=new_K)
    return np.rint(pts_und.reshape(-1, 2)).astype(int)

# ---------- 3. 缩放 + 并排显示 ----------
def show_side_by_side(img1, img2, max_h=360, win_name='distort vs undistort'):
    h, w = img1.shape[:2]
    scale = max_h / h
    new_w = int(w * scale)
    img1 = cv2.resize(img1, (new_w, int(h*scale)), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (new_w, int(h*scale)), interpolation=cv2.INTER_AREA)
    vis = np.hstack([img1, img2])
    cv2.imshow(win_name, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- 4. 主流程 ----------
if __name__ == "__main__":
    img_path  = r"D:\program\li3D-ML\data\calib\camera_intrinsic\data\jiaxing_complete\frame_0.jpg"
    json_path = r"D:\program\li3D-ML\data\calib\camera_intrinsic\data\jiaxing_complete\output\intrinsic.json"

    K, D, h, w = load_intrinsic(json_path)

    img_distort = cv2.imread(img_path)                 # 原始畸变图
    img_undistort = cv2.undistort(img_distort, K, D)   # 整张图矫正

    pts_raw = np.array([[553, 569]])
    pts_und = undistort_points(pts_raw, K, D, img_size=(w, h), alpha=0)

    # 画点
    for x, y in pts_raw:
        cv2.circle(img_distort, (int(x), int(y)), 8, (0, 0, 255), -1)   # 红色
    for x, y in pts_und:
        cv2.circle(img_undistort, (int(x), int(y)), 8, (0, 255, 0), -1) # 绿色

    show_side_by_side(img_distort, img_undistort, max_h=360)