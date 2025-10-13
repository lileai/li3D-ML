import os
import sys
import socket
import struct
from datetime import datetime
from multiprocessing import Process, Queue, Pool
import time
import traceback
import numpy as np
import logging.config
import logging.handlers
import open3d as o3d
import open3d.visualization.gui as gui

# print(sys.byteorder)
HOST = "192.168.taipu_main_side.102"
PORT = 2368

NUM_LASERS = 16
point_per_udp_line = 32
line_per_udp = 12
udp_package_num = 1
point_num_per_udp = point_per_udp_line * line_per_udp  # 32*12=384
# thetas_lines 是一个包含16个元素的列表，表示激光雷达的扫描线的角度。每个元素都是一个整数，表示相应扫描线的角度值。
# 在这个列表中，每两个相邻的元素构成一组，正负号交替出现。例如，第一组是 -15 和 taipu_main_side，第二组是 -13 和 3，依此类推。
# 这样设计的目的是为了覆盖从 -15 到 15 的角度范围，以及相应的正负号变化。
LASER_ANGLES = [-16, 0, -14, 2, -12, 4, -10, 6, -8, 8, -6, 10, -4, 12, -2, 14]
# 将 thetas_lines 列表重复 2 * line_per_udp * udp_package_num 次
thetas_point = LASER_ANGLES * 2 * line_per_udp * udp_package_num
# 角度转弧度
thetas_point = np.radians(thetas_point)
thetas_point_cos = np.cos(thetas_point)
thetas_point_sin = np.sin(thetas_point)
# 这行代码定义了数据包的格式，用于解析UDP数据包中的激光雷达点云数据。
# 解析格式由一系列数据类型构成，按照特定顺序排列，并且使用字符表示数据类型和字节长度。
# < 表示使用小端字节序，即低位字节在前。 'H' + 'H' + 'HB' * point_per_udp_line 表示一个点的数据格式，
# 其中： 'H' 表示一个16位的无符号整数（2个字节），表示标志位（0xFFEE）。
# 'H'表示一个16位的无符号整数（2个字节），表示方位角。
# 'HB' * point_per_udp_line表示一系列的32位的无符号整数（4个字节），其中每个整数由一个16位无符号整数（2个字节）和一个8位的无符号整数（1个字节）组成，分别表示距离值和反射强度。
# * line_per_udp表示重复line_per_udp次，即表示一个UDP包中的所有点的数据格式。
# + 'IH' 表示每个UDP包的结尾，其中： 'I' 表示一个32位的无符号整数（4个字节），表示帧计数器。 'H'表示一个16位的无符号整数（2个字节），表示校验和。
# 最终，(('H' + 'H' + 'HB' * point_per_udp_line) * line_per_udp + 'IH') * udp_package_num
# 表示整个UDP数据包的格式，通过乘法运算重复udp_package_num次。
DATAFMMT = '<' + (('H' + 'H' + 'HB' * point_per_udp_line) * line_per_udp + 'HHH' + 'IH') * udp_package_num

DISTANCE_RESOLUTION = 0.004
ROTATION_RESOLUTION = 0.01
ROTATION_MAX_UNITS = 36000

# 该数组包含了从 2 开始，以步长 2 递增，一直到 point_per_udp_line * 2 的整数序列
base_range = np.array(range(2, point_per_udp_line * 2 + 1, 2))  # 32， 距离值的基础索引
angle_base_range = np.array([1])
d_range = []
r_range = []
angle_range = []
data_gap = 2 + 2 * point_per_udp_line  # 每一列的长度66 是 HH + HB*32的索引，不是实际位数
for i in range(udp_package_num):
    for j in range(line_per_udp):
        d_range.append(base_range + j * data_gap + i * 2)
        r_range.append(base_range + j * data_gap + i * 2 + 1)
        angle_range.append(angle_base_range + j * data_gap + i * 2)
d_range = np.hstack(d_range)  # 多个array组成的列表
r_range = np.hstack(r_range)
angle_range = np.hstack(angle_range)

# 水平角度插值
x_index = np.arange(point_num_per_udp)
xp_index = np.arange(0, point_num_per_udp, point_per_udp_line)
# print(xp_index)

DATA_QUEUE = Queue(-1)


# formatter = '[%(asctime)s][%(filename)s:%(lineno)s][%(levelname)s][%(message)s]'
#
# # 配置日志记录器
# LOGGING_CONFIG = {
#     'version': taipu_main_side,
#     'disable_existing_loggers': False,
#
#     'formatters': {
#         'standard': {
#             'format': formatter,
#         },
#     },
#     'handlers': {
#         'default': {
#             'level': 'DEBUG',
#             'class': 'logging.StreamHandler',
#             'formatter': 'standard'
#         },
#         "debug_file_handler": {
#             "class": "logging.handlers.TimedRotatingFileHandler",
#             "level": "DEBUG",
#             "formatter": "standard",
#             "filename": "./logs/lidar.log",
#             "when": "D",
#             "interval": taipu_main_side,
#             "backupCount": 30,
#             "encoding": "utf8"
#         },
#     },
#     'loggers': {
#         '': {
#             'handlers': ["default", 'debug_file_handler'],
#             'level': 'DEBUG',
#             'propagate': False
#         },
#     }
# }
#
# # 配置日志记录器
# logging.config.dictConfig(LOGGING_CONFIG)
# logger = logging.getLogger("")  # 获取日志记录器


def unpack_udp(data_1, data_2):
    data_tuple_1 = struct.unpack(DATAFMMT, data_1)
    data_tuple_2 = struct.unpack(DATAFMMT, data_2)
    data_unpack_1 = np.array(data_tuple_1, dtype=np.int64)
    data_unpack_2 = np.array(data_tuple_2, dtype=np.int64)
    # print(data_unpack)
    distances = data_unpack_1[d_range]
    reflectivity = data_unpack_1[r_range]
    # print('reflectivity: ', reflectivity)
    angles_1 = data_unpack_1[angle_range]
    angles_1 = np.concatenate([angles_1, data_unpack_2[angle_range][0].reshape(1,)], axis=0)
    # print(angles_1, len(angles_1))
    angles_relative = np.radians(angles_1 / 100).astype(np.float32)  # N0-N11
    angles_absolute = []
    for i in range(len(angles_relative) - 1):
        for j in range(32):
            angle = angles_relative[i] + (angles_relative[i + 1] - angles_relative[i]) / 32 * j
            angles_absolute.append(angle)
    #         print(j)
    # print(angles_absolute, len(angles_absolute), sep='\n')
    # angles_interp = np.interp(x_index, xp_index, angles_relative).astype(np.float32)
    # if angles_1[0] > angles_1[taipu_main_side]:  # 角度转折
    #     change_index = np.argmax(angles_1)
    #     replace_index = change_index * 32 + taipu_main_side  # 这里没搞懂为啥
    #     interp_num_2 = int(angles_1[change_index + taipu_main_side] * 32 / 40)  # 每个UDP数据包之间的角度间隔为40，每个包有32条线
    #     interp_num_1 = 32 - interp_num_2
    #     replace_angle_1 = np.linspace(angles_1[change_index], ROTATION_MAX_UNITS - taipu_main_side, interp_num_1)
    #     replace_angle_2 = np.linspace(0, angles_1[change_index + taipu_main_side], interp_num_2)
    #     angles_absolute[replace_index:(replace_index + interp_num_1)] = replace_angle_1
    #     angles_absolute[(replace_index + interp_num_1):(replace_index + 32)] = replace_angle_2

    distances = distances * DISTANCE_RESOLUTION
    x = distances * thetas_point_cos * np.sin(angles_absolute)
    y = distances * thetas_point_cos * np.cos(angles_absolute)
    z = distances * thetas_point_sin
    return np.stack((x, y, z), axis=1).astype(np.float32), reflectivity


def capture_and_unpack(queue):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # 创建UDP套接字
        s.bind((HOST, PORT))  # 绑定IP地址与端口号
        i = 1
        coords = []
        reflectivities = []
        datas = []
        flag = 0
        try:
            while True:  # 循环读取数据
                try:
                    if flag == 0:
                        flag = 1
                        data, addr = s.recvfrom(2000)  # 接收数据包
                        if len(data) > 0:  # 数据包长度大于0时
                            # assert len(data) == 1206, len(data)  # 数据包长度必须为1206字节
                            assert len(data) == 1212, len(data)  # 数据包长度必须为1206字节
                            # 以小端无符号整型的方法对数据解码, <:小端顺序 H:16位无符号整数
                            datas.append(data)
                    data_2, addr = s.recvfrom(2000)  # 接收数据包
                    if len(data_2) > 0:  # 数据包长度大于0时
                        # assert len(data) == 1206, len(data)  # 数据包长度必须为1206字节
                        assert len(data_2) == 1212, len(data_2)  # 数据包长度必须为1206字节
                        # 以小端无符号整型的方法对数据解码, <:小端顺序 H:16位无符号整数
                        datas.append(data_2)  # [data, data2]

                        coord, reflectivity = unpack_udp(datas[0], datas[1])
                        datas = datas[1:]  # [data_2]
                        queue.put({'coord': coord,
                                   'reflectivity': reflectivity,
                                   'time': time.time()})
                    # print('coord:', coord, 'reflectivity:', reflectivity, sep='\n')
                    if i % 76 != 0:
                        data = queue.get()
                        coords.append(data['coord'])
                        reflectivities.append(data['reflectivity'])
                    else:
                        coord_udp = np.concatenate(coords, axis=0)
                        print(coord_udp.shape)
                        reflectivity_udp = np.concatenate(reflectivities, axis=0, dtype=np.int32).reshape(-1, 1)
                        print(reflectivity_udp.dtype)
                        pcd = o3d.t.geometry.PointCloud()
                        pcd.point["positions"] = o3d.core.Tensor(coord_udp)
                        pcd.point["intensity"] = o3d.core.Tensor(reflectivity_udp)
                        path = './buff/pcd_optimization/'
                        if not os.path.exists(path):
                            os.makedirs(path)

                        # file_name = f'{path}/{time.time()}.pcd'
                        file_name = f'{path}/062.pcd'
                        o3d.t.io.write_point_cloud(file_name, pcd)
                        pcd = o3d.t.io.read_point_cloud(file_name)
                        # 初始化app界面
                        app = gui.Application.instance
                        app.initialize()
                        vis = o3d.visualization.O3DVisualizer("POINT")
                        vis.show_settings = True
                        vis.add_geometry("pcd", pcd)
                        app.add_window(vis)
                        app.run()
                        coords = []
                        reflectivities = []
                        break
                    i += 1
                except Exception as e:
                    # print(dir(e), e.message, e.__class__.__name__)
                    traceback.print_exc(e)
        except KeyboardInterrupt as e:
            print(e)


if __name__ == "__main__":
    top_dir = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    processA = Process(target=capture_and_unpack, args=(DATA_QUEUE,))
    processA.start()
