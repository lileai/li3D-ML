import cv2
import numpy as np
import json
import open3d as o3d

# 读取外参和内参矩阵
def read_calibration_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    T = np.array(data['T']).reshape(4, 4)
    K = np.array(data['K']).reshape(3, 3)
    dist = np.array(data['dist'])
    return T, K, dist

# 将像素坐标转换为相机坐标系
def pixel_to_camera_coords(u, v, depth, K):
    pixel_coords = np.array([u, v, 1])
    camera_coords = np.linalg.inv(K) @ pixel_coords
    camera_coords = camera_coords * depth
    return camera_coords

# 将相机坐标系转换为激光雷达坐标系
def camera_to_lidar_coords(camera_coords, T):
    camera_coords_homogeneous = np.append(camera_coords, 1)
    lidar_coords_homogeneous = T @ camera_coords_homogeneous
    lidar_coords = lidar_coords_homogeneous[:3]
    return lidar_coords

# 读取外参和内参矩阵
T, K, dist = read_calibration_data('D:\program\li3D-ML\scripts\calib\lidar2camera\data\qingshan\calibration_out.json')

# 读取图片并框选区域
def select_roi(image_path):
    image = cv2.imread(image_path)
    roi = cv2.selectROI("Select ROI", image, fromCenter=False)
    cv2.destroyAllWindows()
    return roi

# 主函数
def main(image_path, point_cloud):
    # 读取图片并框选区域
    roi = select_roi(image_path)
    x, y, w, h = roi

    # 假设深度值（从其他数据源获取，如激光雷达点云或深度图）
    depth = 10.0  # 示例深度值

    # 将框选区域的像素坐标转换为相机坐标系
    top_left_camera = pixel_to_camera_coords(x, y, depth, K)
    top_right_camera = pixel_to_camera_coords(x + w, y, depth, K)
    bottom_left_camera = pixel_to_camera_coords(x, y + h, depth, K)
    bottom_right_camera = pixel_to_camera_coords(x + w, y + h, depth, K)

    # 将相机坐标系转换为激光雷达坐标系
    top_left_lidar = camera_to_lidar_coords(top_left_camera, T)
    top_right_lidar = camera_to_lidar_coords(top_right_camera, T)
    bottom_left_lidar = camera_to_lidar_coords(bottom_left_camera, T)
    bottom_right_lidar = camera_to_lidar_coords(bottom_right_camera, T)

    # 将框选区域画到点云中
    import open3d as o3d

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 创建框选区域的线框
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # 上边框
        [4, 5], [5, 7], [7, 6], [6, 4],  # 下边框
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接上下边框
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]

    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([top_left_lidar, top_right_lidar, bottom_left_lidar, bottom_right_lidar,
                                                  top_left_lidar + np.array([0, 0, -1]), top_right_lidar + np.array([0, 0, -1]),
                                                  bottom_left_lidar + np.array([0, 0, -1]), bottom_right_lidar + np.array([0, 0, -1])])
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 可视化点云和线框
    o3d.visualization.draw_geometries([pcd, line_set])

# 示例点云数据（随机生成）
point_cloud = np.asarray(o3d.io.read_point_cloud(r"D:\program\li3D-ML\scripts\calib\lidar2camera\data\qingshan\rename_2025_05_29_10_08_31_000.pcd").points)

# 调用主函数
main(r'D:\program\li3D-ML\scripts\calib\lidar2camera\data\qingshan\frame_01068.jpg', point_cloud)