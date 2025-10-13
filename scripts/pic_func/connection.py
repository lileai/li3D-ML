import cv2
import os
import time
import glob
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

def white_close_bgr(img_bgr, ksize=5, iterations=1):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
# --------- 并查集 ----------
class UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            self.p[y] = x

# --------- 主函数 ----------
def connect_white_regions_bgr_fast(img_bgr, max_gap=None, line_thick=1):
    # 1. 二值化
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(mask) == 0:           # 全黑直接返回
        return img_bgr.copy()

    # 2. 只保留外轮廓 → 点集大幅缩小
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) <= 1:                    # 本来就连在一起
        return img_bgr.copy()

    # 3. 每个轮廓建成 KD-Tree，记录中心点用于快速筛选
    trees, centers, pts_list = [], [], []
    for cnt in contours:
        pts = cnt.squeeze()                   # (N,2)
        if pts.ndim == 1:                     # 单点轮廓
            pts = pts[None, :]
        pts_list.append(pts)
        trees.append(cKDTree(pts))
        centers.append(pts.mean(axis=0))
    centers = np.array(centers)

    # 4. 并查集维护连通性
    uf = UF(len(contours))
    labels_out = np.zeros(mask.shape, np.int32)
    for i, cnt in enumerate(contours):
        cv2.drawContours(labels_out, [cnt], -1, i+1, -1)

    # 5. 贪心连接
    while True:
        # 5.1 找最近的一对“不同根”区域
        best = None
        min_d = np.inf
        alive = [i for i in range(len(contours)) if uf.find(i) == i]
        if len(alive) <= 1:
            break
        # 先粗略过滤：中心距离 > 当前最小距离 的跳过
        for ia, ib in zip(*np.triu_indices(len(alive), 1)):
            i, j = alive[ia], alive[ib]
            if np.linalg.norm(centers[i]-centers[j]) >= min_d:
                continue
            # KD-Tree 精确最近邻
            d, _ = trees[i].query(pts_list[j], k=1)
            dmin = d.min()
            if dmin < min_d:
                min_d = dmin
                best = (i, j)
        if best is None or (max_gap is not None and min_d > max_gap):
            break

        # 5.2 画线 + 合并标签
        i, j = best
        # 再找一次最近两点坐标
        dist_mat = cdist(pts_list[i], pts_list[j])
        idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        p1 = pts_list[i][idx[0]]
        p2 = pts_list[j][idx[1]]
        cv2.line(labels_out, tuple(p1), tuple(p2), i+1, line_thick)
        uf.union(i, j)

        # 5.3 把 j 的轮廓点合并到 i 的树里，更新中心
        new_pts = np.vstack([pts_list[i], pts_list[j]])
        pts_list[i] = new_pts
        trees[i] = cKDTree(new_pts)
        centers[i] = new_pts.mean(axis=0)

    # 6. 转回 BGR
    out = cv2.cvtColor((labels_out > 0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    return out

def connect_x_near_y_edge_weld(img_bgr,
                               x_window=30,   # x 方向“邻近”窗口半径
                               y_gap=20,      # 下-上边缘最大空隙
                               line_thick=1):
    """
    只连接 x 重心相距 ≤ x_window 且上下边缘 y 空隙 ≤ y_gap 的轮廓对。
    每个轮廓都会主动寻找位于自己上方/下方的真正邻居，保证不遗漏。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) <= 1:
        return img_bgr.copy()

    # 1. 每个轮廓的 x 重心 + y 边缘
    info = []
    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim == 1:
            pts = pts[None, :]
        xs, ys = pts[:, 0], pts[:, 1]
        info.append((xs.mean(), ys.min(), ys.max()))  # (cx, y_min, y_max)

    # 2. 按 x 重心排序，滑动窗口找“x 邻近”对
    order = sorted(range(len(info)), key=lambda k: info[k][0])
    uf = UF(len(contours))
    canvas = mask.copy()

    for i in range(len(order)):
        cx_i, y_min_i, y_max_i = info[order[i]]
        # 向后扫，直到 x 差 > x_window
        for j in range(i + 1, len(order)):
            cx_j, y_min_j, y_max_j = info[order[j]]
            if cx_j - cx_i > x_window:
                break
            if uf.find(order[i]) == uf.find(order[j]):
                continue

            # 3. 统一上下关系：让 lower 始终指下方轮廓
            if y_min_j > y_max_i:          # j 在 i 上方
                upper, lower = j, i
                y_gap_val = y_min_j - y_max_i
            elif y_min_i > y_max_j:        # i 在 j 上方
                upper, lower = i, j
                y_gap_val = y_min_i - y_max_j
            else:                          # 有重叠
                y_gap_val = 0

            if y_gap_val > y_gap:
                continue

            # 4. 画线：下方轮廓上边缘最近点 → 上方轮廓下边缘最近点
            pts_low = contours[order[lower]].squeeze()
            pts_upp = contours[order[upper]].squeeze()
            if pts_low.ndim == 1:
                pts_low = pts_low[None, :]
            if pts_upp.ndim == 1:
                pts_upp = pts_upp[None, :]
            dist_mat = cdist(pts_low, pts_upp)
            r, c = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            p1, p2 = tuple(pts_low[r]), tuple(pts_upp[c])
            cv2.line(canvas, p1, p2, 255, line_thick)
            uf.union(order[i], order[j])

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
# ----------------------------------------------------------

def process_folder_save_imgs(img_dir: str,
                             out_dir: str,
                             ksize=7, iters=10,
                             max_gap=50, line_thick=3):
    """
    批量处理并保存图片，文件名与原图相同，仅更换输出目录
    """
    os.makedirs(out_dir, exist_ok=True)          # 确保输出目录存在
    pics = glob.glob(os.path.join(img_dir, '*.*'))
    if not pics:
        raise FileNotFoundError(f'No images in {img_dir}')

    print('Start processing ...')
    tik = time.time()
    for idx, fp in enumerate(pics, 1):
        img = cv2.imread(fp)
        if img is None:
            continue

        # 处理管道
        frame = white_close_bgr(img, ksize=ksize, iterations=iters)
        frame = connect_white_regions_bgr_fast(frame, max_gap=max_gap, line_thick=line_thick)
        frame = connect_x_near_y_edge_weld(frame, x_window=25, y_gap=200, line_thick=line_thick)

        # 保存：文件名不变，目录换成 out_dir
        out_path = os.path.join(out_dir, os.path.basename(fp))
        cv2.imwrite(out_path, frame)

        if idx % 50 == 0:
            print(f'Processed {idx}/{len(pics)}  ...')

    print(f'Done!  Total time: {time.time()-tik:.2f}s  -> {out_dir}')


# ---------------------- main ----------------------
if __name__ == '__main__':
    src_folder = r"D:\program\li3D-ML\data\SuQian\0925_out"  # 原图目录
    dst_folder = r"D:\program\li3D-ML\data\SuQian\0925_out_processed"  # 结果目录
    process_folder_save_imgs(src_folder,
                             dst_folder,
                             ksize=5, iters=5,
                             max_gap=50, line_thick=4)