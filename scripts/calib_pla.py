"""
点云平面标定
"""
import os
import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui


def fit_plane_and_rotate(pcd):
    """
    使用Open3D拟合点云的平面，并将点云旋转到与该平面平行。
    :param pcd: Open3D的PointCloud对象。
    :return: 旋转后的点云和旋转矩阵。
    """
    # 拟合平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # 归一化法向量

    # 计算旋转矩阵
    z_axis = np.array([0, 0, 1])
    axis = np.cross(normal, z_axis)
    axis = axis / np.linalg.norm(axis)  # 归一化旋转轴
    angle = np.arccos(np.dot(normal, z_axis))

    # 使用罗德里格斯公式计算旋转矩阵
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # 旋转点云
    rotated_points = np.asarray(pcd.points) @ rotation_matrix

    # 更新点云对象
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)
    rotated_pcd.colors = pcd.colors  # 保留颜色信息（如果有）

    # 提取拟合平面上的点
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # 将拟合平面上的点染成红色

    return rotated_pcd, rotation_matrix, inlier_cloud

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


if __name__ == '__main__':
    # 读取PCD文件
    pcd_path = r"D:\program\li3D-ML\data\Taipu-main\pcd\ship_1756512816.2025824_nan_111340_-8.570954613387585.pcd"  # 替换为你的PCD文件路径
    pcd = o3d.io.read_point_cloud(pcd_path)
    l = split_filename_from_path(pcd_path)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:int(l[-2])])

    # 拟合平面并旋转点云
    rotated_pcd, rotation_matrix, inlier_cloud = fit_plane_and_rotate(pcd)

    print("旋转矩阵:\n", rotation_matrix.tolist())

    # 可视化原始点云、旋转后的点云和拟合平面上的点云
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(
        f"Point Cloud Visualization",
        width=1660, height=900)
    vis.show_skybox(False)
    vis.show_settings = True
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None
    vis.add_geometry("Original Point Cloud", pcd)
    vis.add_geometry("Rotated Point Cloud", rotated_pcd)
    vis.add_geometry("Fitted Plane Points", inlier_cloud)
    app.add_window(vis)
    app.run()
    o3d.io.write_point_cloud(f'rotation.pcd', rotated_pcd, print_progress=False)