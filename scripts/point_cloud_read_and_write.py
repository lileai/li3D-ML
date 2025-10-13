# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 19:35
# @Author  : Leii
# @File    : point_cloud_read_and_write.py
# @Code instructions: 
# ------------------------------------------------------------------------
import numpy as np
import datetime
import open3d as o3d


if __name__ == '__main__':
    filename = '000450.pcd'
    dataset = 'customer_data'
    pcd_file = f"../data/{dataset}/{filename}"
    pcd = o3d.t.io.read_point_cloud(f"../data/{dataset}/{filename}")
    pcd_intensity = pcd.point["intensity"]  # 强度
    pcd_points = pcd.point["positions"]  # 坐标
    pcd_ring = pcd.point["ring"]  # 环数
    pcd_time = pcd.point["time"]  # 时间戳

    pcd_intensity = pcd_intensity[:, :].numpy()  # 转换为数组类型
    pcd_points = pcd_points[:, :].numpy()  # 转换为数组类型
    pcd_ring = pcd_ring[:, :].numpy()  # 转换为数组类型
    pcd_time = pcd_time[:, :].numpy()  # 转换为数组类型
    scan = np.concatenate((pcd_points, pcd_intensity), axis=-1)
    print(scan.shape)
    print(scan[0:5])

    print(pcd_points.shape)
    print(pcd_points[0:5])

    print(pcd_intensity.shape)
    print(pcd_intensity[0:5])

    print(pcd_ring.shape)
    print(pcd_ring[0:5])

    print(pcd_time.shape)
    print(pcd_time[0:20])
