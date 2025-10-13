import os
import time
import copy
import json
import open3d as o3d
import numpy as np
import glob
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui


def load_point_clouds(filename, file_extention=".pcd"):
    """
    从文件夹中读取点云
    :param filename: 单个文件路径
    :param file_extention: 文件后缀
    :return: 单帧点云
    """
    if file_extention == ".pcd":
        pcd = o3d.io.read_point_cloud(filename)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    elif file_extention == ".bin":
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(scan[:, :3]))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    return pcd


def get_color_from_colormap(index, num_points, colormap_name='Set1'):
    """
    从指定的色彩映射中获取颜色
    :param index: 当前点云的索引
    :param num_points: 点云总数
    :param colormap_name: 色彩映射名称
    :return: RGB 颜色值
    """
    colormap = plt.get_cmap(colormap_name)
    return colormap(index / num_points)[:3]  # 只取 RGB 值，忽略透明度


def right_multiply_transform(pcd, transformation):
    """
    将点云的每个点右乘变换矩阵
    :param pcd: 点云
    :param transformation: 变换矩阵
    :return: 变换后的点云
    """
    points = np.asarray(pcd.points)
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = homogeneous_points @ transformation.T
    pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])
    return pcd


if __name__ == '__main__':
    start_time = time.time()

    # taipu_main_side. 加载多个传感器的相同帧数据
    data_path = "./icp_data/"  # 请改成你实际存在的路径
    pcd_files = glob.glob(os.path.join(data_path, '*.pcd'))
    pcds = []

    # 预定义的变换矩阵
    T = [
        [
            [
                0.9911806788524264,
                0.13207456084366695,
                -0.010824613045663475,
                0.026759051410783184
            ],
            [
                -0.1323840123972415,
                0.9905381418358328,
                -0.0361754451254258,
                0.03577044886196966
            ],
            [
                0.005944336064077078,
                0.037289407964837054,
                0.9992868281541529,
                0.041269505232307956
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        [
            [
                0.9944797328255031,
                -0.10459516183717175,
                -0.00836140655462663,
                0.021297410569509116
            ],
            [
                0.1045370612796214,
                0.9944955859471604,
                -0.007108611019792442,
                -0.07485628866832537
            ],
            [
                0.009058908230939643,
                0.006195292718339648,
                0.9999397754513975,
                0.055216207730583625
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    ]

    # 从文件名中提取广播码，并保存对应的逆矩阵到 T.json_files 文件中
    T_dict = {}
    for i, file_name in enumerate(pcd_files):
        print(file_name)
        pcd = load_point_clouds(file_name, file_extention=".pcd")
        pcds.append(pcd)

        # 提取广播码
        bc = os.path.basename(file_name).split('-')[0]
        # 计算逆矩阵
        T_inv = np.linalg.inv(T[i])
        # 保存广播码和逆矩阵
        T_dict[bc] = T_inv.tolist()

    # 保存到 JSON 文件
    with open('../T.json', 'w') as f:
        json.dump(T_dict, f, indent=4)

    print("Transform points and display")
    combined_pcd = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        bc = os.path.basename(pcd_files[point_id]).split('-')[0]
        T_inv = np.array(T_dict[bc])
        pcds[point_id] = right_multiply_transform(pcds[point_id], T_inv)
        combined_pcd += pcds[point_id]

    # 可视化
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(
        "icp",
        width=1660, height=900)
    vis.show_skybox(False)
    vis.show_settings = True
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    vis.set_background(background_color, None)

    # 保存变换矩阵到JSON文件
    o3d.io.write_point_cloud(f"../../data/icp_data/ICP/icp_curr.pcd", combined_pcd)
    print(time.time() - start_time)
    vis.add_geometry("combined_pcd", combined_pcd)
    app.add_window(vis)
    app.run()