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
HOST = "192.168.taipu_main_side.100"
PORT = 2368

NUM_LASERS = 16
point_per_udp_line = 32
line_per_udp = 12
udp_package_num = 1
point_num_per_udp = point_per_udp_line * line_per_udp  # 32*12=384
# thetas_lines 是一个包含16个元素的列表，表示激光雷达的扫描线的角度。每个元素都是一个整数，表示相应扫描线的角度值。
# 在这个列表中，每两个相邻的元素构成一组，正负号交替出现。例如，第一组是 -15 和 taipu_main_side，第二组是 -13 和 3，依此类推。
# 这样设计的目的是为了覆盖从 -15 到 15 的角度范围，以及相应的正负号变化。
LASER_ANGLES = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
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
DATAFMMT = '<' + (('H' + 'H' + 'HB' * point_per_udp_line) * line_per_udp + 'IH') * udp_package_num

DISTANCE_RESOLUTION = 0.002
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

DATA_QUEUE = Queue(-1)

formatter = '[%(asctime)s][%(filename)s:%(lineno)s][%(levelname)s][%(message)s]'

# 配置日志记录器
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': formatter,
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        "debug_file_handler": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "./logs/lidar.log",
            "when": "D",
            "interval": 1,
            "backupCount": 30,
            "encoding": "utf8"
        },
    },
    'loggers': {
        '': {
            'handlers': ["default", 'debug_file_handler'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}

# 配置日志记录器
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("")  # 获取日志记录器


def unpack_udp(data):
    data_tuple = struct.unpack(DATAFMMT, data)
    data_unpack = np.array(data_tuple, dtype=np.int64)
    print(data_unpack)
    distances = data_unpack[d_range]
    reflectivity = data_unpack[r_range]
    # print('reflectivity: ', reflectivity)
    angles = data_unpack[angle_range]
    print(angles)
    angles = np.radians(angles / 100).astype(np.float32)
    angles_interp = np.interp(x_index, xp_index, angles).astype(np.float32)
    if angles[0] > angles[1]:  # 角度转折
        change_index = np.argmax(angles)
        replace_index = change_index * 32 + 1  # 这里没搞懂为啥
        interp_num_2 = int(angles[change_index + 1] * 32 / 40)  # 每个UDP数据包之间的角度间隔为40，每个包有32条线
        interp_num_1 = 32 - interp_num_2
        replace_angle_1 = np.linspace(angles[change_index], ROTATION_MAX_UNITS - 1, interp_num_1)
        replace_angle_2 = np.linspace(0, angles[change_index + 1], interp_num_2)
        angles_interp[replace_index:(replace_index + interp_num_1)] = replace_angle_1
        angles_interp[(replace_index + interp_num_1):(replace_index + 32)] = replace_angle_2

    distances = distances * DISTANCE_RESOLUTION
    x = distances * thetas_point_cos * np.sin(angles_interp)
    y = distances * thetas_point_cos * np.cos(angles_interp)
    z = distances * thetas_point_sin
    return np.stack((x, y, z), axis=1).astype(np.float32), reflectivity


def capture_and_unpack(queue):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # 创建UDP套接字
        s.bind((HOST, PORT))  # 绑定IP地址与端口号
        i = 1
        coords = []
        reflectivities = []
        try:
            while True:  # 循环读取数据
                try:
                    data, addr = s.recvfrom(2000)  # 接收数据包
                    if len(data) > 0:  # 数据包长度大于0时
                        assert len(data) == 1206, len(data)  # 数据包长度必须为1206字节
                        # 以小端无符号整型的方法对数据解码, <:小端顺序 H:16位无符号整数
                        coord, reflectivity = unpack_udp(data)
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
                        path = './buff'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        file_name = f'{path}/{time.time()}.pcd'
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
