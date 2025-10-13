import open3d as o3d
import numpy as np
import os
import open3d.visualization.gui as gui
import glob

def apply_rotation_matrix_to_point_cloud(pcd, R):
    """
    对点云中的每个点应用旋转矩阵
    """
    points = np.asarray(pcd.points)
    rotated_points = (R @ points.T).T  # 对每个点应用旋转矩阵
    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(rotated_points)
    return pcd_rotated

if __name__ == '__main__':
    filename = 'pointcloud_2025_06_26_09_07_38.pcd'
    o3d.utility.random.seed(42)
    dataset = 'out'
    directory_path = f"../data/0618/{dataset}"
    out_path = f"{directory_path}/ship_out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 批量读取指定路径下的pcd文件名
    pcd_files = glob.glob(f'{directory_path}/*.pcd', recursive=True)
    n = 374
    for i, pcd_file in enumerate(pcd_files[n:], start=n):
        pcd = o3d.io.read_point_cloud(pcd_file, format="pcd")
        points = np.asarray((pcd.points))
        print(f"第{i}帧点云的点数为：{points.shape[0]}")
        # 读取矫正参数矫正pcd
        rotation_matrix = np.loadtxt(f"{directory_path}/qiaolin_rotation_matirx.txt")
        pcd = apply_rotation_matrix_to_point_cloud(pcd, rotation_matrix)
        # # 去掉除船以外的其他点云
        index = np.where(((points[:, 0] < 0) & (points[:, 1] >= 160)) | ((points[:, 0] >= 0) & (points[:, 1] >= 100)))[0]
        ship_cloud = pcd.select_by_index(index)
        # ui
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer(f"{pcd_file}", width=1660, height=1000)
        vis.show_skybox(False)
        vis.show_settings = True
        # 设置背景颜色为黑色
        background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
        vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None

        # set_viewer = True if os.path.exists(f'./view/{dataset}_viewpoint.json_files') else False
        set_viewer = False
        if set_viewer:  # 自定义可视化初始视角参数
            param = o3d.io.read_pinhole_camera_parameters(f'./view/{dataset}_viewpoint.json')
            extrinsic_matrix, intrinsic_matrix = param.extrinsic, param.intrinsic
            intrinsic_matrix, height, width = intrinsic_matrix.intrinsic_matrix, intrinsic_matrix.height, intrinsic_matrix.width
            vis.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)

        vis.add_geometry("pcd", pcd)
        vis.add_geometry("ship_cloud", ship_cloud)
        app.add_window(vis)
        app.run()
        # 选择性删除
        user_input = input(f"是否需要删除{pcd_file}(y/n)").strip().lower()
        if user_input == 'y':
            os.remove(pcd_file)
            print(f"已删除文件: {pcd_file}")
        elif user_input == 'n':
            print("文件保留。")
        else:
            print("输入无效，文件保留。")