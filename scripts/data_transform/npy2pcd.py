import os
import json
import open3d as o3d
import numpy as np
from tqdm import tqdm


def load_extrinsic(path):
    """读取 4x4 外参矩阵"""
    with open(path) as f:
        d = json.load(f)
        T = []
        for item in d["lidar_list"]:
            T.append(np.array(item["calibration_matrix"]).reshape(4, 4))
        return T


def apply_rotation_matrix_to_point_cloud(points, R):
    """
    对 (N,3) 点云应用 3×3 旋转 或 4×4 齐次矩阵
    返回 (N,3)
    """
    points = np.asarray(points, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)

    if R.shape == (3, 3):
        return (R @ points.T).T
    elif R.shape == (4, 4):
        # 升维 → 齐次坐标
        h_pts = np.hstack([points, np.ones((points.shape[0], 1))])
        h_rot = (R @ h_pts.T).T
        return h_rot[:, :3]  # 降回 3D
    else:
        raise ValueError("R 必须是 (3,3) 或 (4,4)")


def npy2pcd(input_path, output_path, **kwargs):
    """
    读取.npy文件，应用变换矩阵，保存为.pcd文件
    """
    try:
        # 读取变换矩阵
        T = kwargs.get("T", np.eye(4))
        # 读取.npy为ndarray
        arr = np.load(input_path)
        arr = np.asarray(arr)
        # 初始化open3d的tensor对象
        pcd = o3d.t.geometry.PointCloud()
        if arr.shape[1] == 3:  # 只有坐标
            points = arr[:, :3]
        elif arr.shape[1] == 4:  # 带有强度信息
            points = arr[:, :3]
            intensity = arr[:, 3]
            pcd.point["intensity"] = o3d.core.Tensor(intensity.reshape(-1, 1))
        # 应用变换矩阵到points
        transform_points = apply_rotation_matrix_to_point_cloud(points, T)
        pcd.point["positions"] = o3d.core.Tensor(transform_points)
        output_path = os.path.splitext(output_path)[0] + '_TRANSFORM.pcd'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.t.io.write_point_cloud(output_path, pcd)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_directory(input_dir, output_dir, func=None, **kwargs):
    """
    批量处理目录中的.npy文件
    """
    # 传入必要的变换矩阵
    T = kwargs.get("T", [np.eye(4)])
    # 收集所有图像文件路径
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.npy')):
                image_files.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    loops = tqdm(image_files, desc="Processing pcds", unit="pcd")
    for input_path in loops:
        # 忽略 detail 文件夹
        if "detail" in input_path:
            continue

        # 获取文件名的结尾数字
        file_name = os.path.basename(input_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        index = 0  # 默认使用 T[0]
        if "_" in file_name_without_extension:
            try:
                index = int(file_name_without_extension.split("_")[-1])
            except ValueError:
                print(f"Using default transformation matrix for {file_name} as the index is not a valid number.")

        # 选择对应的变换矩阵
        if index < len(T):
            T_matrix = T[index]
        else:
            print(f"Using default transformation matrix for {file_name} as index {index} is out of range for T.")
            T_matrix = T[0]

        # 构建输出路径
        relative_path = os.path.relpath(os.path.dirname(input_path), input_dir)
        output_path = os.path.join(output_dir, relative_path, os.path.basename(input_path))

        # 调用处理函数
        saved = func(input_path, output_path, T=T_matrix)

        # 如果未保存，输出提示信息
        if not saved:
            tqdm.write(f"Skipped: {os.path.basename(input_path)}")

    # 合并同一个文件夹下的所有点云
    merge_point_clouds(output_dir)


def merge_point_clouds(output_dir):
    """
    合并同一个文件夹下的所有点云，忽略detail文件夹
    """
    for root, dirs, files in os.walk(output_dir):
        # 忽略detail文件夹
        if 'detail' in dirs:
            dirs.remove('detail')

        for dir in dirs:
            if '.pcd' in dirs:
                dirs.remove('.pcd')
            pcd_dir = os.path.join(root, dir)
            if os.path.exists(pcd_dir):
                pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('_TRANSFORM.pcd')]
                if pcd_files:
                    merged_pcd = o3d.t.geometry.PointCloud()
                    base_name = os.path.basename(pcd_files[0]).split('_')[1]  # 获取基础文件名
                    for i, pcd_file in enumerate(pcd_files):
                        pcd_path = os.path.join(pcd_dir, pcd_file)
                        pcd = o3d.t.io.read_point_cloud(pcd_path)
                        if i == 0:
                            merged_pcd = pcd
                        else:
                            merged_pcd.point["positions"] = o3d.core.Tensor(np.concatenate(
                                ((merged_pcd.point["positions"].numpy(), pcd.point["positions"].numpy()))))
                            merged_pcd.point["intensity"] = o3d.core.Tensor(np.concatenate(
                                ((merged_pcd.point["intensity"].numpy(), pcd.point["intensity"].numpy()))))
                    merge_dir = os.path.join(root, ".pcd")
                    os.makedirs(merge_dir, exist_ok=True)
                    merged_output_path = os.path.join(merge_dir, f"{base_name}_MERGE.pcd")
                    o3d.t.io.write_point_cloud(merged_output_path, merged_pcd)
                    print(f"Merged point cloud saved to {merged_output_path}")


# 示例调用
if __name__ == "__main__":
    data_path = r"../../data"
    dataset = r"JaXing\D\04_12_04_50_6a5fe126-a722-4f78-bff0-037402311c03_7\recurrent_dir"
    input_directory = rf'{data_path}/{dataset}/original_npy'  # 替换为你的输入目录路径
    json_path = r'../json_files/lidar_config_jiaxing.json'  # 替换为你的输入json文件的路径
    output_directory = rf'{data_path}/{dataset}/original_npy'  # 替换为你的输出目录路径
    T = load_extrinsic(json_path)
    process_directory(input_directory, output_directory, func=npy2pcd, T=T)