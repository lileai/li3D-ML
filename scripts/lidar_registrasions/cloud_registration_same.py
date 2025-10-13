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


def pairwise_registration(source, target, max_correspondence_distance_coarse=0.2,
                          max_correspondence_distance_fine=0.01):
    """
    应用点到面的两两配准
    :param source: 源点云
    :param target: 目标点云
    :param max_correspondence_distance_coarse: 粗配准对应点的最大距离
    :param max_correspondence_distance_fine: 精配准对应点的最大距离
    :return: 变换矩阵，变换矩阵相关信息
    """
    T = [
        [
            -0.9224554573919408,
            0.380880690464898,
            -0.06329161681205131,
            124.6270000000018
        ],
        [
            -0.3778632649837468,
            -0.9242423158860277,
            -0.054731110909004385,
            54.299999999999855
        ],
        [
            -0.07934281381148628,
            -0.026571434972438857,
            0.9964931895100759,
            6.817000000000009
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ]
    voxel_size = 0.02
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)
    # 粗配准
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance_coarse,
        init=np.asarray(T),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # 精配准
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance_fine,
        init=icp_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine, icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    """
    完全配准（多个传感器的相同帧）
    :param pcds: 输入的点云列表，每个传感器一个点云
    :param max_correspondence_distance_coarse: 粗配准对应点的最大距离
    :param max_correspondence_distance_fine: 精配准对应点的最大距离
    :return: 姿态图
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)

    reference_pcd = pcds[0]
    for source_id in range(1, n_pcds):
        transformation_icp, information_icp = pairwise_registration(
            reference_pcd,
            pcds[source_id],
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine)
        print(f"Build o3d.pipelines.registration.PoseGraph for sensor {source_id}")
        # 添加节点和边
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(transformation_icp)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(0,
                                                     source_id,
                                                     transformation_icp,
                                                     information_icp,
                                                     uncertain=False))
    return pose_graph


def save_transformations_to_json(pose_graph, output_file="transformations.json_files"):
    """
    将每个点云的配准矩阵写入一个JSON文件中
    :param pose_graph: 姿态图
    :param output_file: 输出文件路径
    """
    transformations = []
    for i, node in enumerate(pose_graph.nodes):
        transformations.append({
            f"sensor_{i}": node.pose.tolist()
        })

    with open(output_file, 'w') as f:
        json.dump(transformations, f, indent=4)


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


if __name__ == '__main__':
    start_time = time.time()

    # taipu_main_side. 加载多个传感器的相同帧数据
    data_path = "../../data/icp_data/"  # 请改成你实际存在的路径
    pcd_files = glob.glob(os.path.join(data_path, '*.pcd'))
    pcds = []
    for i, file_name in enumerate(pcd_files):
        print(file_name)
        pcd = load_point_clouds(file_name, file_extention=".pcd")
        pcds.append(pcd)

    for point_id in range(len(pcds)):
        color = get_color_from_colormap(point_id, len(pcds), colormap_name='Set1')
        pcds[point_id].paint_uniform_color(color)
    pcds_copy = copy.deepcopy(pcds)

    # 4. 多个传感器的相同帧配准
    print("Full registration ...")
    voxel_size = 1.5
    max_correspondence_distance_coarse = voxel_size * 25
    max_correspondence_distance_fine = voxel_size * 2.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:

        pose_graph = full_registration(pcds,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    print("Transform points and display")
    combined_pcd = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
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
    save_transformations_to_json(pose_graph, output_file=f"../../data/icp_data/ICP/transformations.json")
    o3d.io.write_point_cloud(f"../../data/icp_data/ICP/icp.pcd", combined_pcd)
    print(time.time() - start_time)
    for i, pcd in enumerate(pcds_copy):
        vis.add_geometry(f"pcd_{i}", pcd)
    vis.add_geometry("combined_pcd", combined_pcd)
    app.add_window(vis)
    app.run()