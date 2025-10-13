import open3d as o3d
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm   # 进度条，可选

# ---------------- 拷贝自你的实现 ----------------
def build_box(points):
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)
    vertices = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z]
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(vertices)
    box.lines  = o3d.utility.Vector2iVector(edges)
    box.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(edges))])
    return box

def split_filename_from_path(file_path):
    file_name_with_extension = os.path.basename(file_path)
    file_name, _ = os.path.splitext(file_name_with_extension)
    parts = file_name.split('_')
    return [p for p in parts if p]
# ----------------------------------------------

def render_pcd_box_once(pcd, box, width=1600, height=990, preview=False):
    # ---------- 1. 计算 L/W/H ----------
    pts = np.asarray(box.points)
    min_b = pts.min(axis=0)
    max_b = pts.max(axis=0)
    L, W, H = max_b - min_b          # 长、宽、高
    text = f"SHIP: L={L:.2f}  W={W:.2f}  H={H:.2f}"

    # ---------- 2. 预览 ----------
    if preview:
        frame_lidar = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        o3d.visualization.draw_geometries([pcd, box, frame_lidar],
                                          window_name="preview",
                                          width=width, height=height)

    # ---------- 3. 离线渲染 ----------
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)
    vis.add_geometry(box)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(
        r"D:\program\li3D-ML\scripts\view\icp_data_viewpoint.json")
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    img = (np.asarray(img) * 255).astype(np.uint8)

    # ---------- 4. 把 L/W/H 写到图像 ----------
    img = cv2.putText(img, text, (20, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return img

def pcd_files_to_video(root_dir, out_video='output.mp4', fps=10):
    # 收集所有 pcd
    pcd_files = glob.glob(f'{root_dir}/*.pcd', recursive=True)
    if not pcd_files:
        raise RuntimeError('未找到任何 pcd 文件，请检查路径')

    # 预读一帧拿 shape
    temp_pcd = o3d.io.read_point_cloud(pcd_files[0])
    temp_img = render_pcd_box_once(temp_pcd, build_box(np.asarray(temp_pcd.points)))
    h, w = temp_img.shape[:2]

    # 初始化视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    for pcd_file in tqdm(pcd_files, desc='rendering'):
        parts = split_filename_from_path(pcd_file)
        keep_num = int(parts[-2])          # 按你的规则截取点数
        pcd = o3d.io.read_point_cloud(pcd_file)
        pts = np.asarray(pcd.points)[:keep_num]
        pcd.points = o3d.utility.Vector3dVector(pts)

        box = build_box(pts)
        img = render_pcd_box_once(pcd, box, width=w, height=h)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # OpenCV 需要 BGR

    vw.release()
    print('视频已保存至:', os.path.abspath(out_video))

if __name__ == '__main__':
    root = '../data/deck_pcd'
    pcd_files_to_video(root, out_video='deck_pcd.mp4', fps=10)