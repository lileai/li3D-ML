import json, os, numpy as np
import glob
import open3d as o3d
import open3d.visualization.gui as gui


def load_json(path):
    with open(path) as f:
        return np.array(json.load(f).get("extrinsic", []), dtype=float)


def compute_and_save(root=r"../../../data/calib/lidar2lidar/data"):
    T = [[0.9782723722514219, -0.14451163910356643, 0.1486591801846623, 0.0],
         [0.14189075377425603, 0.9894821484548106, 0.028144126965062688, 0.0],
         [-0.15116275891551054, -0.006439258719192405, 0.9884879140708795, 0.0], [0.0, 0.0, 0.0, 1.0]]
    # 读
    m_3_5 = load_json(os.path.join(root, "3-5.json"))
    m_5_2 = load_json(os.path.join(root, "5-2.json"))
    m_2_1 = load_json(os.path.join(root, "2-1.json"))
    m_1_4 = load_json(os.path.join(root, "1-4.json"))
    m_4_0 = load_json(os.path.join(root, "4-0.json"))

    # 算
    m_3_0 = T @ m_4_0 @ m_1_4 @ m_2_1 @ m_5_2 @ m_3_5
    m_5_0 = T @ m_4_0 @ m_1_4 @ m_2_1 @ m_5_2
    m_2_0 = T @ m_4_0 @ m_1_4 @ m_2_1
    m_1_0 = T @ m_4_0 @ m_1_4
    m_4_0 = T @ m_4_0
    m_0_0 = T @ np.identity(4)

    # 存
    out_path = os.path.join(root, "lidar2lidar_extrinsic.json")
    with open(out_path, "w") as f:
        json.dump(
            {k: v.tolist() for k, v in {
                "lidar3_to_lidar0": m_3_0,
                "lidar5_to_lidar0": m_5_0,
                "lidar2_to_lidar0": m_2_0,
                "lidar1_to_lidar0": m_1_0,
                "lidar4_to_lidar0": m_4_0,
                "lidar0_to_lidar0": m_0_0
            }.items()},
            f, indent=4
        )
    print(f"saved -> {out_path}")
    map_pcd = \
        {3: m_3_0,
         5: m_5_0,
         2: m_2_0,
         1: m_1_0,
         4: m_4_0,
         0: m_0_0, }
    return map_pcd


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


if __name__ == "__main__":
    map_pcd = compute_and_save()
    data_path = r"../../../data/calib/lidar2lidar/data"  # 请改成你实际存在的路径
    pcd_files = glob.glob(os.path.join(data_path, '*.pcd'))
    for pcd_file in pcd_files:
        l = split_filename_from_path(pcd_file)
        lidar_id = int(l[1])  # 保证是整数
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd.transform(map_pcd[lidar_id])
        # 如果后面还要用，把变换后的点云放到新 dict
        map_pcd[lidar_id] = pcd
        z_point = np.asarray(pcd.points)
        z_point[:,2] = 0
        pcd.points = o3d.utility.Vector3dVector(z_point)

    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(
        width=1660, height=900)
    vis.show_skybox(False)
    vis.show_settings = True
    # 设置背景颜色为黑色
    background_color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # RGBA颜色值，范围从0到1
    vis.set_background(background_color, None)  # 第二个参数是背景图像，这里不需要，所
    for key, value in map_pcd.items():
        vis.add_geometry(f"{key}", value)

    app.add_window(vis)
    app.run()
