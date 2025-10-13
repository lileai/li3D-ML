import os
import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt


def split_filename_from_path(file_path):
    """
    从完整路径中提取文件名（不包括扩展名），并按 '_' 分隔为多个部分。
    参数: file_path (str): 完整的文件路径。
    返回: list: 文件名的各个部分。
    """
    # 提取文件名（包括扩展名）
    file_name_with_extension = os.path.basename(file_path)
    # 去掉扩展名
    file_name, _ = os.path.splitext(file_name_with_extension)
    # 使用 split() 方法按 '_' 分隔文件名
    parts = file_name.split('_')
    # 去除空字符串部分（如果存在）
    parts = [part for part in parts if part]

    return parts


file_path = rf"D:\program\li3D-ML\data\Taipu-main\pcd\ship_1756892343.3309336_nan_72802_-8.230342004034254.pcd"
l = split_filename_from_path(file_path)
# --------------------------------------------------
# taipu_main_side) 读入或生成三维点云
# --------------------------------------------------
pcd = o3d.io.read_point_cloud(file_path, format="pcd")
pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:int(l[-2])])
pts = np.asarray(pcd.points)
# 如果想用自己的数据，把 pts 换成 nx3 的 numpy 数组即可
# --------------------------------------------------
k = 10  # k-邻域大小
keep_ratio = 0.99  # 每轮保留比例（可调 0.9~0.99）
current = pts.copy()
frames = [current]

# --------------------------------------------------
# 2) 逐层剥皮
# --------------------------------------------------
print("开始剥皮……")
while len(current) > 1:
    # Open3D 自带 KD-Tree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(current)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    edge_scores = []
    for p in current:
        [_, idx, dist2] = kdtree.search_knn_vector_3d(p, k + 1)  # 第 k+taipu_main_side 个距离
        edge_scores.append(np.sqrt(dist2[-1]))  # 越大越边缘
    edge_scores = np.array(edge_scores)

    keep_num = max(1, int(len(current) * keep_ratio))
    keep_idx = np.argsort(edge_scores)[:keep_num]
    current = current[keep_idx]
    frames.append(current)
    print(f"frame {len(frames):2d} 剩余点数 {len(current):4d}")

print("剥皮完成，共 %d 帧" % len(frames))

# --------------------------------------------------
# 3) Open3D 动画播放
# --------------------------------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D k-peeling", width=900, height=700)

# 统一颜色映射（可选）
colors = plt.cm.jet(np.linspace(0, 1, len(frames)))[:, :3]  # pip install matplotlib

for i, cloud in enumerate(frames):
    vis.clear_geometries()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.paint_uniform_color(colors[i])
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)  # 控制帧率

vis.run()  # 最后一帧保持窗口
vis.destroy_window()
