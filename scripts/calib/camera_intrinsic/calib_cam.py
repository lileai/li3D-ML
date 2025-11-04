#!/usr/bin/env python3
"""
单目相机标定脚本（带可视化 & 实时输出 & PDF 报告）
用法:
    python calib_cam.py --img_dir ./data/hik --pdf_report
"""
import cv2, numpy as np, json, os, shutil, argparse
from pathlib import Path
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.model_selection import ShuffleSplit


# -------------------- 参数解析 --------------------
def parse_args():
    data_path = r"../../../data/calib/camera_intrinsic/data/jiaxing_capture"
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default=data_path, type=str, help="标定图像文件夹")
    parser.add_argument('--out_dir', default=rf"{data_path}/output", help="输出的路径")
    parser.add_argument('--w', default=12, type=int, help="标定板横向角点数")
    parser.add_argument('--h', default=10, type=int, help="标定板纵向角点数")
    parser.add_argument('--square_size', default=52, type=float, help="方格边长(mm)")
    parser.add_argument('--min_images', default=15, type=int, help="最少有效图像数")
    parser.add_argument('--max_reproj_err', default=0.7, type=float, help="最大允许重投影误差")
    parser.add_argument('--pdf_report', default=True, help="标定后生成 PDF 报告")
    return parser.parse_args()


# -------------------- 工具函数 --------------------
def collect_images(root):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif')
    files = []
    for ext in exts:
        files.extend(Path(root).rglob(ext))
    return sorted(files)


def compute_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_brightness(gray):
    return np.mean(gray)


def validate_corner_layout(corners, pattern_size):
    return len(corners) == pattern_size[0] * pattern_size[1]


# -------------------- 图像筛选 & 角点检测 --------------------
def select_and_show(images, pattern_size, args):
    selected, objpoints, imgpoints = [], [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-6)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= args.square_size
    total, valid = 0, 0
    for p in images:
        total += 1
        img = cv2.imread(str(p))
        if img is None:
            print(f"无法读取 {p.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpness = compute_sharpness(gray)
        brightness = compute_brightness(gray)
        if sharpness < 50:
            print(f"清晰度低: {p.name} (sharpness={sharpness:.1f})")
            continue
        if brightness < 40 or brightness > 220:
            print(f"曝光异常: {p.name} (brightness={brightness:.1f})")
            continue
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not ret:
            print(f"❌ 未检测到标定板: {p.name}")
            continue
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if not validate_corner_layout(corners, pattern_size):
            print(f"❌ 角点布局异常: {p.name}")
            continue
        selected.append(p)
        objpoints.append(objp)
        imgpoints.append(corners)
        valid += 1
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, ret)
        cv2.putText(vis, f"Sharp: {sharpness:.1f}, Bright: {brightness:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Calibration - Use ESC to exit", vis)
        if cv2.waitKey(500) == 27:
            break
    cv2.destroyAllWindows()
    print(f"\n📊 总计 {total} 张 | 有效 {valid} 张 | 选中 {len(selected)} 张")
    img_size = gray.shape[::-1] if selected else None
    return selected, objpoints, imgpoints, img_size


# -------------------- 重投影误差 --------------------
def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, D):
    errors = []
    for i in range(len(objpoints)):
        proj_pts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], proj_pts, cv2.NORM_L2) / len(proj_pts)
        errors.append(err)
    return np.array(errors)


# -------------------- 标定主流程 --------------------
def calibrate(selected, objpoints, imgpoints, img_size, square_size, args):
    new_obj = [o * square_size for o in objpoints]
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        new_obj, imgpoints, img_size, None, None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-10))
    mean_error = compute_reprojection_errors(new_obj, imgpoints, rvecs, tvecs, K, D)
    print(f"初始平均重投影误差（MAE）: {mean_error.mean():.4f} 像素")
    if len(selected) > args.min_images:
        filtered_indices = [i for i, err in enumerate(mean_error) if err < args.max_reproj_err]
        if len(filtered_indices) >= args.min_images:
            objpoints = [objpoints[i] for i in filtered_indices]
            imgpoints = [imgpoints[i] for i in filtered_indices]
            print(f"重新标定，使用 {len(filtered_indices)} 张高质量图像")
            ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-10))
    return K, D, ret, rvecs, tvecs, objpoints, imgpoints


# -------------------- 保存 & 去畸变 --------------------
def save_selected(selected, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for p in selected:
        shutil.copy2(str(p), out_dir)


def undistort(selected, K, D, out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for p in selected:
        img = cv2.imread(str(p))
        und = cv2.undistort(img, K, D)
        cv2.imwrite(str(Path(out_dir) / p.name), und)


def save_intrinsic(K, D, img_size, file='intrinsic.json'):
    # 获取文件的父目录路径
    dir_path = os.path.dirname(file)
    # 如果父目录路径不存在，则创建它
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data = {"K": K.tolist(), "dist": D[0].tolist(), "img_size": [img_size[0], img_size[1]]}
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"内参已保存到 {file}")


# -------------------- PDF 报告 --------------------
def plot_reproj_error_heatmap(objpoints, imgpoints, rvecs, tvecs, K, D, img_size):
    all_proj_pts, all_img_pts = [], []
    for i in range(len(objpoints)):
        proj_pts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        all_proj_pts.extend(proj_pts.reshape(-1, 2))
        all_img_pts.extend(imgpoints[i].reshape(-1, 2))
    all_proj_pts = np.array(all_proj_pts)
    all_img_pts = np.array(all_img_pts)
    errors = np.linalg.norm(all_proj_pts - all_img_pts, axis=1)

    plt.rcParams['font.family'] = 'SimSun'  # Windows 宋体
    plt.figure(figsize=(6, 4))
    plt.scatter(all_img_pts[:, 0], all_img_pts[:, 1], c=errors, cmap='jet', s=8)
    plt.colorbar(label='误差（像素）')
    plt.title("重投影误差热力图")
    plt.xlabel("X 像素")
    plt.ylabel("Y 像素")
    plt.gca().invert_yaxis()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


def robust_test_focal_variation(selected, objpoints, imgpoints, img_size):
    ss = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    fs = []
    for train_idx, _ in ss.split(selected):
        obj_tr = [objpoints[i] for i in train_idx]
        img_tr = [imgpoints[i] for i in train_idx]
        _, K, _, _, _ = cv2.calibrateCamera(obj_tr, img_tr, img_size, None, None)
        fs.append(K[0, 0])
    return np.std(fs) / np.mean(fs) * 100


def generate_pdf_report(K, D, img_size, selected, errors, rvecs, tvecs, objpoints, imgpoints, path):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # 注册中文字体
    try:
        pdfmetrics.registerFont(TTFont('song', 'simsun.ttc'))
        song = 'song'
    except:
        pdfmetrics.registerFont(TTFont('song', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'))
        song = 'song'

    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4

    # 标题
    c.setFont(song, 18)
    c.drawString(50, h - 50, "相机标定报告")

    # 正文
    c.setFont(song, 11)
    y = h - 90
    c.drawString(50, y, f"●  图像数量：{len(selected)} 张")
    y -= 20
    c.drawString(50, y, f"●  平均重投影误差：{np.mean(errors):.3f} 像素")
    y -= 20
    c.drawString(50, y, f"●  误差标准差：{np.std(errors):.3f} 像素")
    y -= 20
    c.drawString(50, y, f"●  最大误差：{np.max(errors):.3f} 像素")
    y -= 20
    c.drawString(50, y, f"●  焦距 fx：{K[0, 0]:.2f} 像素")
    y -= 20
    c.drawString(50, y, f"●  主点偏移：({abs(K[0, 2] - img_size[0] / 2):.1f}, {abs(K[1, 2] - img_size[1] / 2):.1f}) 像素")
    y -= 20
    # ✅ 畸变合理性：多留 50 像素安全区
    c.drawString(50, y, f"●  畸变系数合理性：{'合格' if all(abs(d) < 0.5 for d in D[0][:4]) else '异常'}")
    y -= 50          # ← 安全间距
    c.drawString(50, y, "●  重投影误差热力图：")
    y -= 10          # 再留一点空白

    # ✅ 先生成图片，再贴图（顺序靠后）
    heatmap_buf = plot_reproj_error_heatmap(objpoints, imgpoints, rvecs, tvecs, K, D, img_size)
    c.drawImage(ImageReader(heatmap_buf), 50, y - 260, width=400, height=260)
    y -= 280

    # 鲁棒性
    focal_var = robust_test_focal_variation(selected, objpoints, imgpoints, img_size)
    c.drawString(50, y, f"●  鲁棒性测试（焦距波动）：{focal_var:.2f} %")
    if focal_var > 3:
        c.drawString(50, y - 20, "⚠️  警告：可能存在过拟合，建议增加图像数量或多样性！")

    y -= 40  # 与上一段留点空

    # ------------------------------------------------------------------
    # 1. 计算 5 项子分（0–100）
    # ------------------------------------------------------------------
    n_img = len(selected)
    mean_e = np.mean(errors)
    std_e = np.std(errors)
    max_e = np.max(errors)
    foc_var = focal_var

    # 1.1 图像数量分（10 张→100 分，线性插值，最多 100）
    score_n = min(100., n_img * 10.)

    # 1.2 平均误差分（0.05 px→100 分，0.5 px→0 分，线性）
    score_mean = max(0., 100 * (0.5 - mean_e) / 0.45)

    # 1.3 误差标准差分（0.02 px→100 分，0.3 px→0 分）
    score_std = max(0., 100 * (0.3 - std_e) / 0.28)

    # 1.4 最大误差分（0.1 px→100 分，2 px→0 分）
    score_max = max(0., 100 * (2.0 - max_e) / 1.9)

    # 1.5 鲁棒性分（焦距波动 0 %→100 分，5 %→0 分）
    score_var = max(0., 100 * (5.0 - foc_var) / 5.0)

    # ------------------------------------------------------------------
    # 2. 加权综合（权重可自己调）
    # ------------------------------------------------------------------
    weights = np.array([0.10, 0.30, 0.20, 0.20, 0.20])  # 顺序对应上面 5 项
    scores = np.array([score_n, score_mean, score_std, score_max, score_var])
    total = float(np.dot(weights, scores))

    # ------------------------------------------------------------------
    # 3. 画到 PDF
    # ------------------------------------------------------------------
    c.setFont(song, 14)
    c.drawString(50, y, "●  综合评分：")
    y -= 30

    # 画一个醒目色块
    box_w, box_h = 120, 50
    c.setFillColorRGB(0.13, 0.55, 0.13)  # 深绿
    c.rect(50, y - box_h, box_w, box_h, fill=1)

    # 在色块中央写白字
    c.setFillColorRGB(1, 1, 1)
    c.setFont(song, 28)
    c.drawCentredString(50 + box_w / 2, y - box_h + 12, f"{total:.0f}")

    c.save()
    print(f"📄 中文 PDF 报告已保存：{path}")


# -------------------- main --------------------
def main():
    args = parse_args()
    images = collect_images(Path(args.img_dir))
    if not images:
        print("未找到任何图片")
        return
    selected, objp, imgp, img_size = select_and_show(images, (args.w, args.h), args)
    if len(selected) < 5:
        print("有效图片不足 5 张，标定终止")
        return
    K, D, err, rvecs, tvecs, objp, imgp = calibrate(selected, objp, imgp, img_size, args.square_size, args)
    print(f"📏 重投影误差(RMS): {err:.4f} 像素")
    save_selected(selected, f'{args.img_dir}/selected')
    undistort(selected, K, D, f'{args.img_dir}/undistorted')
    save_intrinsic(K, D, img_size, f'{args.out_dir}/intrinsic.json')

    if args.pdf_report:
        errors = compute_reprojection_errors(objp, imgp, rvecs, tvecs, K, D)
        generate_pdf_report(K, D, img_size, selected, errors, rvecs, tvecs, objp, imgp,
                            path=f'{args.out_dir}/calibration_report.pdf')


if __name__ == '__main__':
    main()
