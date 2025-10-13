"""
频域点云滤波
"""
import os
import time
import open3d as o3d
import numpy as np
from scipy.ndimage import gaussian_filter
import open3d.visualization.gui as gui
from scipy.spatial import cKDTree

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

def spatial_mask_islands(
        xyz: np.ndarray,
        voxel_size: float = 0.01,
        sigma: float = 1.0,
        thresh_pct: float = 5.0
    ) -> np.ndarray:
    """
    用空域 3-D 高斯卷积去除点云中的孤岛点。

    参数
    ----
    xyz : np.ndarray, shape=(N,3)
        点云坐标
    voxel_size : float
        体素边长
    sigma : float
        高斯核标准差（体素单位）
    thresh_pct : float
        保留密度高于该百分位的体素（指非零体素内的百分位）

    返回
    ----
    mask : np.ndarray, shape=(N,), dtype=bool
        True 表示保留，False 表示删除
    """

    # taipu_main_side. 体素栅格
    min_corner = xyz.min(axis=0) - voxel_size
    max_corner = xyz.max(axis=0) + voxel_size
    grid_shape = np.ceil((max_corner - min_corner) / voxel_size).astype(int)

    indices = ((xyz - min_corner) / voxel_size).astype(int)
    indices = np.clip(indices, 0, grid_shape - 1)

    density = np.zeros(grid_shape, dtype=np.float32)
    np.add.at(density, (indices[:, 0], indices[:, 1], indices[:, 2]), 1.0)

    # 2. 空域高斯卷积
    density_smooth = gaussian_filter(density, sigma=sigma)

    # 3. 阈值
    non_zero = density_smooth[density_smooth > 0]
    if non_zero.size == 0:           # 极端情况：空点云
        return np.ones(len(xyz), dtype=bool)

    thresh = np.percentile(non_zero, thresh_pct)
    valid_voxel = density_smooth > thresh

    # 4. 体素映射回点
    mask = valid_voxel[indices[:, 0], indices[:, 1], indices[:, 2]]
    return mask

def statistical_outlier(points, nb_neighbors=30, std_ratio=1.0, is_inliers=False):
    """统计滤波去除离群点"""
    # 构建 KDTree
    tree = cKDTree(points)

    # 计算每个点的平均距离
    distances, _ = tree.query(points, k=nb_neighbors)
    mean_distances = np.mean(distances, axis=1)

    # 计算标准差
    mean_distance = np.mean(mean_distances)
    stddev = np.std(mean_distances)

    # 找到离群点
    inliers = np.abs(mean_distances - mean_distance) < std_ratio * stddev
    if is_inliers:
        return inliers
    else:
        return points[inliers]

def points2pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

if __name__ == '__main__':
    start_time = time.time()
    # file_path = rf"D:\program\li3D-ML\data\Taipu-main\pcd\ship_1756892343.3309336_nan_72802_-8.230342004034254.pcd"
    # file_path = rf"D:\program\li3D-ML\data\Taipu-main\pcd\ship_1756514075.6290686_-10.944531440734863_63572_-10.513974053519112.pcd"
    # file_path = rf"D:\program\li3D-ML\data\Taipu-main\pcd\ship_1756888142.1510503_nan_40146_-9.274276796976725.pcd"
    file_path = rf"D:\program\li3D-ML\data\Taipu-main\08_11_27_09_a68af434-eeaf-4132-a04b-f4b42d0cf378_3\detail\deck_pcd\ship_1757302046.0840204_nan_18192_-10.936608632405598.pcd"
    l = split_filename_from_path(file_path)
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:int(l[-2])])
    points = np.asarray(pcd.points)
    voxel_size = 2.0
    sigma = 1.01
    thresh_pct = 65
    mask = spatial_mask_islands(points, voxel_size=voxel_size, sigma=sigma, thresh_pct=thresh_pct)
    filter_points = points[mask]
    filter_points = statistical_outlier(filter_points, nb_neighbors=10, std_ratio=5.0)
    filter_pcd = points2pcd(filter_points)
    print(time.time() - start_time)
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Mesh", width=1024, height=768)
    vis.show_skybox(False)
    vis.show_settings = True
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None
    vis.add_geometry("pcd", pcd)
    vis.add_geometry("filter_pcd", filter_pcd)
    app.add_window(vis)
    app.run()
