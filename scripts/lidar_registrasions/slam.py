def pairwise_registration(source, target, max_correspondence_distance_coarse=0.2,
                          max_correspondence_distance_fine=0.1):
    """
    应用点到面的两两配准
    :param source: 源点云
    :param target: 目标点云
    :param max_correspondence_distance_coarse: 粗配准对应点的最大距离
    :param max_correspondence_distance_fine: 精配准对应点的最大距离
    :return: 变换矩阵，变换矩阵相关信息
    """
    print("应用点到面的ICP")
    # --------------------------------------粗配准-------------------------------------------
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse,  # 源点云、目标点云、对应点之间的最大距离
        init=np.identity(4),                                 # 初始化的单位矩阵
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())  # 设置配准方法为：点到面
    # -------------------------------精配准(各参数含义同上)-----------------------------------
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        init=icp_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    # --------------------------------配准转换矩阵的相关信息-----------------------------------
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, 0.1, icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    """
    完全配准
    :param pcds:输入的点云
    :param max_correspondence_distance_coarse: 粗配准对应点的最大距离
    :param max_correspondence_distance_fine: 精配准对应点的最大距离
    :return:姿态图
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
# -----------------------------------odometry case----------------------------------------
            if target_id == source_id + 1:
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
# --------------------------------loop closure case--------------------------------------
            else:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph
