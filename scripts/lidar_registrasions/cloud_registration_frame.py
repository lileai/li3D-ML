import os
import glob
import yaml
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui

def get_file_extension(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower()  # 将后缀名转为小写，方便比较

def load_point_clouds(pcd_files, file_extention=".pcd"):
    """
    从文件夹中读取点云
    :param pcd_files: 点云文件路径列表
    :return: 点云列表
    """
    pcds = []
    if file_extention == ".pcd":
        for filename in pcd_files:
            pcd = o3d.io.read_point_cloud(filename)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcds.append(pcd)
    elif file_extention == ".bin":
        for filename in pcd_files:
            scan = np.fromfile(filename, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(scan[:, :3]))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcds.append(pcd)
    return pcds

def pairwise_registration(source, target, max_correspondence_distance_coarse=0.02,
                          max_correspondence_distance_fine=0.1):
    """
    应用点到面的两两配准
    :param source: 源点云
    :param target: 目标点云
    :param max_correspondence_distance_coarse: 粗配准对应点的最大距离
    :param max_correspondence_distance_fine: 精配准对应点的最大距离
    :return: 变换矩阵，变换矩阵相关信息
    """
    # 粗配准
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance_coarse,  # 源点云、目标点云、对应点之间的最大距离
        init=np.identity(4),  # 初始化的单位矩阵
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())  # 设置配准方法为：点到面
    # 精配准
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance_fine,
        init=icp_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    # 配准转换矩阵的相关信息
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, 0.1, icp_fine.transformation)

    # 计算配准精度
    inlier_rmse = icp_fine.inlier_rmse

    return transformation_icp, information_icp, inlier_rmse

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    """
    完全配准（激光里程计模式：仅配准相邻帧）
    :param pcds: 输入的点云
    :param max_correspondence_distance_coarse: 粗配准对应点的最大距离
    :param max_correspondence_distance_fine: 精配准对应点的最大距离
    :return: 姿态图
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(1, n_pcds):
        target_id = source_id - 1
        transformation_icp, information_icp, inlier_rmse = pairwise_registration(
            pcds[target_id],
            pcds[source_id],
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine)
        print("Build o3d.pipelines.registration.PoseGraph")
        # 连续帧
        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(target_id,
                                                     source_id,
                                                     transformation_icp,
                                                     information_icp,
                                                     uncertain=False))
    return pose_graph

def save_transformations_to_yaml(pose_graph, output_file="transformations.yaml"):
    """
    将每个点云的配准矩阵写入一个YAML文件中
    :param pose_graph: 姿态图
    :param output_file: 输出文件路径
    """
    transformations = []
    for i, node in enumerate(pose_graph.nodes):
        transformations.append({
            f"point_cloud_{i}": node.pose.tolist()
        })

    with open(output_file, 'w') as f:
        yaml.dump(transformations, f, default_flow_style=False)

def get_color_from_colormap(index, num_points, colormap_name='viridis'):
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
    dataset = 'icp_data'
    directory_path = f"../../data/{dataset}"
    out_path = f"{directory_path}/ICP"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 批量读取指定路径下的pcd文件名
    # pcd_files = glob.glob(f'{directory_path}/*.pcd', recursive=True)
    pcd_files = glob.glob(f'{directory_path}/*.pcd', recursive=True)
    pcds = load_point_clouds(pcd_files=pcd_files, file_extention=".pcd")
    # 全局配准
    print("Full registration ...")
    voxel_size = 0.02
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    pose_graph = full_registration(pcds,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,  # 对应点的阈值
        edge_prune_threshold=0.25,  # 修剪异常边缘的阈值
        reference_node=0)  # 被视为全局空间的节点ID
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    print("Transform points and display")
    combined_pcd = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        # print(pose_graph.nodes[point_id].pose)
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        color = get_color_from_colormap(point_id, len(pcds), colormap_name='viridis')
        pcds[point_id].paint_uniform_color(color)
        combined_pcd += pcds[point_id]

    # 保存变换矩阵到YAML文件
    save_transformations_to_yaml(pose_graph, output_file=f"{out_path}/transformations.yaml")
    # 可视化
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(
        "icp",
        width=1660, height=900)
    vis.show_skybox(False)
    vis.show_settings = True
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所以传入None
    for i, pcd in enumerate(pcds):
        vis.add_geometry(f"pcd_{i}", pcd)
    vis.add_geometry("combined_pcd", combined_pcd)
    app.add_window(vis)
    app.run()