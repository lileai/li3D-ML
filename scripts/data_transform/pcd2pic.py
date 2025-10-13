import open3d as o3d
import cv2
import numpy as np
import open3d.visualization.gui as gui


# 读取PCD文件
def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


# 沿x轴旋转点云
def rotate_point_cloud(pcd, angle_degree):
    # 将角度转换为弧度
    angle_radian = np.deg2rad(angle_degree)
    # 创建旋转矩阵
    rotation_matrix = pcd.get_rotation_matrix_from_xyz((angle_radian, 0, 0))
    # 应用旋转矩阵
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    return pcd


def pcd2pic(pcd, width=1000, height=1000):
    # 将点云转换为 NumPy 数组
    points = np.asarray(pcd.points)

    # 提取 x, y, z 坐标
    x = points[:, 0]
    y = points[:, 1]

    # 归一化 x 和 y 坐标到 [0, width-1] 和 [0, height-1]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_norm = (x - x_min) / (x_max - x_min) * (width - 1)
    y_norm = (y - y_min) / (y_max - y_min) * (height - 1)

    # 创建一个空白图像
    img = np.zeros((height, width), dtype=np.uint8)

    # 将点云投影到图像上
    for i in range(len(x_norm)):
        x_int = int(x_norm[i])
        y_int = int(y_norm[i])
        if 0 <= x_int < width and 0 <= y_int < height:
            img[y_int, x_int] = int(255)

    return img


# 主函数
if __name__ == "__main__":
    # 读取PCD文件
    file_path = r"D:\program\li3D-ML\data\Taipu-main\pcd\ori_1756783155.5894063.pcd"  # 替换为你的PCD文件路径
    pcd = read_pcd(file_path)

    # 指定旋转角度
    angle_degree = -25  # 替换为你想要的旋转角度
    pic = pcd2pic(pcd)
    # 使用 OpenCV 显示图像
    cv2.imshow("Point Cloud Image", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rotated_pcd = rotate_point_cloud(pcd, angle_degree)
    pic = pcd2pic(rotated_pcd)
    # 使用 OpenCV 显示图像
    cv2.imshow("Point Cloud Image", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
