import logging
import os
import socket
import time
import traceback
from collections import deque
from dataclasses import dataclass
import datetime
from logging.handlers import RotatingFileHandler
from multiprocessing import Queue, Lock
import threading

import numpy as np
import open3d as o3d

# 配置参数
HOST = "192.168.101.178"  # 雷达IP
PORT = 2368  # 雷达端口
NUMS = 1  # NUMS帧保存一次pcd


# 配置日志
def setup_logging(log_path="./lidar_log"):
    logger = logging.getLogger("lidar_logger")
    logger.setLevel(logging.DEBUG)  # 设置总日志级别为DEBUG，以捕获所有日志

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 文件日志处理器
    file_handler = RotatingFileHandler(
        f"{log_path}/lidar.log",
        maxBytes=1024 * 1024 * 5,
        backupCount=5,
        encoding="utf-8",
        mode="a"
    )
    file_handler.setLevel(logging.DEBUG)  # 文件处理器记录DEBUG及以上级别的日志
    file_formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler.setFormatter(file_formatter)

    # 终端日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 终端处理器仅记录INFO及以上级别的日志
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 清除可能存在的旧处理器，避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 阻止日志传播到根日志记录器，避免重复日志
    logger.propagate = False

    return logger


# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger("lidar_logger")


@dataclass
class LidarData:
    ID: int
    H_angle: float
    V_angle: float
    Distance: float
    Intensity: int
    X: float
    Y: float
    Z: float
    Mtimestamp_nsce: int


# def negative2positive(angle):
#     return int(angle) % 360 if angle >= 0 else 360 - int(abs(angle)) % 360
def negative2positive(value):
    value_t = int(round(value * 1000))
    return value_t % 360000 if value_t >=0 else (value_t % -360000) + 360000

class GetLidarData_LS:
    def __init__(self):
        self.isQuit = False
        self.allDataValue = deque()  # 使用 deque 替代 Queue
        self.m_mutex = Lock()
        self.m_HorizontalPoints = 0
        self.m_StackFrame = 0 if NUMS == 1 else NUMS
        self.m_fV_AngleAcc = 0.0025
        self.m_fH_AngleAcc = 0.01
        self.m_fDistanceAcc = 0.001
        self.m_fPutTheMirrorOffAngle = np.array([1.5, -0.5, 0.5, -1.5])
        self.m_DeadZoneOffset = 6.37
        self.m_UTC_Time = None
        self.lastAllTimestamp = 0
        self.lidar_EchoModel = 0
        self.PointIntervalT_ns = 0
        self.count = 0
        self.m_messageCount = 0
        self.allTimestamp = 0
        self.m_DistanceIsNotZero = 0
        self.m_DistanceThreshold = 20
        self.LidarPerFrameDatePrt_Get = deque()
        self.PointCloudLastData = deque()
        self.tempPointCloud = deque()
        # self.cosAngleValue = np.cos(np.deg2rad(np.arange(361)))
        # self.sinAngleValue = np.sin(np.deg2rad(np.arange(361)))
        # 预计算角度的余弦和正弦值
        FF = np.arange(360000)
        angle_rad = FF / 1000.0 * np.pi / 180.0
        self.cosAngleValue = np.cos(angle_rad)
        self.sinAngleValue = np.sin(angle_rad)
        self.mLidaFilterParamDisplayValue = {
            'mChannelVector': np.ones(4, dtype=int),
            'mMin_Distance': 0,
            'mMax_Distance': 1000,
            'mMin_Intensity': 0,
            'mMax_Intensity': 255,
            'mMin_HanleValue': -360,
            'mMax_HanleValue': 360,
            'mMin_VanleValue': -180,
            'mMax_VanleValue': 180
        }
        self.buffer = bytearray()

    def clearQueue(self, q):
        q.clear()  # 使用 deque 的 clear 方法

    def LidarRun(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((HOST, PORT))  # 示例地址和端口
            s.settimeout(5)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

            logger.info("开始监听...")
            DELIMITER = b'\xff\xaa\xbb\xcc\xdd'
            DELIMITER_LEN = 5
            while True:
                if self.isQuit:
                    logger.info("清空队列")
                    self.clearQueue(self.allDataValue)
                    break

                try:
                    data, addr = s.recvfrom(2048)
                    while True:
                        self.buffer.extend(data)
                        start = self.buffer.find(DELIMITER)
                        if start == -1:
                            # 没找到帧头，保留数据等待下一包
                            if len(self.buffer) > 8192:  # 防止 buffer 膨胀
                                del self.buffer[:len(self.buffer) - 4096]
                            break
                        # 查找下一个帧头或 buffer 结尾
                        print("找到当前帧分割标志为", start)
                        next_start = self.buffer.find(DELIMITER, start + DELIMITER_LEN)
                        if next_start == -1:
                            # 当前帧未完整接收
                            if len(self.buffer) - start > 4096:  # 超长帧？清理
                                self.buffer = self.buffer[start + DELIMITER_LEN:]
                            break
                        print("找到下一帧的帧分割标志为", next_start)
                        # 提取完整帧：从当前帧头到下一个帧头之前
                        frame = self.buffer[start:next_start]
                        with self.m_mutex:
                            self.allDataValue.append(frame)
                        # 移除已处理数据
                        self.buffer = self.buffer[next_start:]
                        print(len(self.buffer), next_start)
                except socket.timeout:
                    logger.error("接收超时，继续监听...")
                except Exception as e:
                    logger.error(f"捕获错误: {str(e)}")
                    traceback.print_exc()

                while not len(self.allDataValue) == 0:
                    data = self.allDataValue.popleft()
                    # 解码逻辑与 C++ 代码类似，但使用 Python 的语法和函数
                    if data[0] in [0x00, 0xa5] and data[1] == 0xff and data[2] == 0x00 and data[3] == 0x5a:
                        self.m_HorizontalPoints = (data[184] << 8) + data[185]
                        if data[231] in [64, 65]:
                            self.m_StackFrame = 2
                        elif data[231] == 192:
                            self.m_StackFrame = 1
                            self.m_fV_AngleAcc = 0.01
                            self.m_fDistanceAcc = 0.004
                            self.m_fPutTheMirrorOffAngle = np.array([1.5, 0.5, -0.5, -1.5])
                            self.m_DeadZoneOffset = 10.82
                        else:
                            self.m_fDistanceAcc = 0.001
                            self.m_fH_AngleAcc = 0.01
                            self.m_fV_AngleAcc = 0.0025
                            self.m_StackFrame = 1
                            self.m_DeadZoneOffset = 6.37
                            self.m_fPutTheMirrorOffAngle = np.array([1.5, -0.5, 0.5, -1.5])
                        majorVersion = data[1202]
                        minorVersion1 = data[1203] // 16
                        minorVersion2 = data[1203] % 16
                        if majorVersion > 1 or (majorVersion == 1 and minorVersion1 > 1):
                            self.m_fV_AngleAcc = 0.0025
                        else:
                            self.m_fV_AngleAcc = 0.01
                        continue

                    timestamp_nsce = (
                            (data[1200] << 24) +
                            (data[1201] << 16) +
                            (data[1202] << 8) +
                            data[1203]
                    )

                    if data[1194] == 0xff:
                        timestamp_s = (
                                (data[1195] * 4294967296) +
                                (data[1196] << 24) +
                                (data[1197] << 16) +
                                (data[1198] << 8) +
                                data[1199]
                        )
                        self.allTimestamp = timestamp_s * 1000000000 + timestamp_nsce
                    else:
                        self.m_UTC_Time = {
                            'year': data[1194] + 2000,
                            'month': data[1195],
                            'day': data[1196],
                            'hour': data[1197],
                            'minute': data[1198],
                            'second': data[1199]
                        }
                        # 校验时间数据的合法性
                        try:
                            dt = datetime.datetime(
                                self.m_UTC_Time['year'], self.m_UTC_Time['month'], self.m_UTC_Time['day'],
                                self.m_UTC_Time['hour'], self.m_UTC_Time['minute'], self.m_UTC_Time['second']
                            )
                            timestamp_s = int(dt.timestamp())
                            self.allTimestamp = timestamp_s * 1000000000 + timestamp_nsce
                        except ValueError as e:
                            logger.error(f"时间数据错误: {e}")
                            continue

                    if data[1204] == 192 and self.m_HorizontalPoints != 0:
                        self.m_StackFrame = 1
                        self.m_fV_AngleAcc = 0.01
                        self.m_fDistanceAcc = 0.004
                        self.m_fPutTheMirrorOffAngle = np.array([1.5, 0.5, -0.5, -1.5])
                        self.m_DeadZoneOffset = 10.82

                    self.lidar_EchoModel = data[1205]
                    if self.lidar_EchoModel in [0x02, 0x12]:
                        self.handleDoubleEcho(data)
                    else:
                        self.handleSingleEcho(data)
                    self.lastAllTimestamp = self.allTimestamp

                else:
                    time.sleep(0.001)

    def handleSingleEcho(self, data):
        self.PointIntervalT_ns = (self.allTimestamp - self.lastAllTimestamp) / 149.0
        for i in range(0, 1192, 8):
            if data[i] == 0xff and data[i + 1] == 0xaa and data[i + 2] == 0xbb and data[i + 3] == 0xcc and data[
                i + 4] == 0xdd:
                self.count += 1
            if self.count == NUMS:
                logger.info(f"捕获到共{NUMS}帧信息")
                logger.debug(f"m_StackFrame:{self.m_StackFrame}")
                if self.m_StackFrame > 1:
                    self.tempPointCloud = self.LidarPerFrameDatePrt_Get.copy()
                    self.LidarPerFrameDatePrt_Get = self.PointCloudLastData + self.LidarPerFrameDatePrt_Get
                    self.PointCloudLastData = self.tempPointCloud
                self.send_lidar_data()
                self.count = 0
                self.m_messageCount = 0
                continue
            if len(self.LidarPerFrameDatePrt_Get) > 3000000:
                if self.m_messageCount == 1:
                    continue
                logger.error("Frame synchronization failure!!!")
                self.m_messageCount = 1
                continue
            if self.count == 0:
                tDistance = float((data[i + 4] << 16) + (data[i + 5] << 8) + data[i + 6])
                self.m_DistanceIsNotZero = self.m_DistanceIsNotZero if self.m_DistanceIsNotZero > 30 else (
                    self.m_DistanceIsNotZero + 1 if abs(tDistance - 0) > 1e-8 else self.m_DistanceIsNotZero)
                if tDistance <= 0:
                    continue

                tfAngle_H = float((data[i] << 8) + data[i + 1])
                if tfAngle_H > 32767:
                    tfAngle_H -= 65536
                tfAngle_H *= self.m_fH_AngleAcc

                tTempAngle = data[i + 2]
                tChannelID = tTempAngle >> 6
                tSymmbol = (tTempAngle >> 5) & 0x01
                if tSymmbol == 1:
                    iAngle_V = (data[i + 2] << 8) + data[i + 3]
                    tfAngle_V = float(iAngle_V | 0xc000)
                    if tfAngle_V > 32767:
                        tfAngle_V -= 65536
                else:
                    iAngle_Hight = tTempAngle & 0x3f
                    tfAngle_V = float((iAngle_Hight << 8) + data[i + 3])
                tfAngle_V *= self.m_fV_AngleAcc

                tDistance *= self.m_fDistanceAcc
                intensity = data[i + 7]

                if tChannelID >= len(self.mLidaFilterParamDisplayValue['mChannelVector']):
                    logger.error("id无效")
                    continue
                if self.mLidaFilterParamDisplayValue['mChannelVector'][tChannelID] == 0:
                    logger.error("通道未启用")
                    continue
                if not self.mLidaFilterParamDisplayValue['mMin_Distance'] <= tDistance <= \
                       self.mLidaFilterParamDisplayValue['mMax_Distance']:
                    logger.error(f"距离超出范围")
                    continue
                if not self.mLidaFilterParamDisplayValue['mMin_Intensity'] <= intensity <= \
                       self.mLidaFilterParamDisplayValue['mMax_Intensity']:
                    logger.error("反射强度超出范围")
                    continue
                m_point = self.XYZ_calculate(tChannelID, tfAngle_H, tfAngle_V, tDistance, -1)

                if not self.mLidaFilterParamDisplayValue['mMin_HanleValue'] <= tfAngle_H <= \
                       self.mLidaFilterParamDisplayValue['mMax_HanleValue']:
                    logger.error("水平角度超出范围")
                    continue
                if not self.mLidaFilterParamDisplayValue['mMin_VanleValue'] <= tfAngle_V <= \
                       self.mLidaFilterParamDisplayValue['mMax_VanleValue']:
                    logger.error("垂直角度超出范围")
                    continue

                m_DataT = LidarData(
                    ID=tChannelID,
                    H_angle=tfAngle_H,
                    V_angle=tfAngle_V,
                    Distance=tDistance,
                    Intensity=intensity,
                    X=m_point['x1'],
                    Y=m_point['y1'],
                    Z=m_point['z1'],
                    Mtimestamp_nsce=self.allTimestamp - self.PointIntervalT_ns * (148 - i // 8)
                )
                self.LidarPerFrameDatePrt_Get.append(m_DataT)

    def handleDoubleEcho(self, data):
        self.PointIntervalT_ns = (self.allTimestamp - self.lastAllTimestamp) / 99
        for i in range(0, 1188, 12):
            if data[i] == 0xff and data[i + 1] == 0xaa and data[i + 2] == 0xbb and data[i + 3] == 0xcc and data[
                i + 4] == 0xdd:
                self.count += 1
            if self.count == NUMS:
                logger.info(f"捕获到共{NUMS}帧信息")
                if self.m_StackFrame > 1:
                    self.tempPointCloud = self.LidarPerFrameDatePrt_Get.copy()
                    self.LidarPerFrameDatePrt_Get = self.PointCloudLastData + self.LidarPerFrameDatePrt_Get
                    self.PointCloudLastData = self.tempPointCloud
                self.send_lidar_data()
                self.count = 0
                self.m_messageCount = 0
                continue
            if len(self.LidarPerFrameDatePrt_Get) > 3000000:
                if self.m_messageCount == 1:
                    continue
                logger.error("Frame synchronization failure!!!")
                self.m_messageCount = 1
                continue
            if self.count == 0:
                tDistance = (data[i + 4] << 16) + (data[i + 5] << 8) + data[i + 6]
                self.m_DistanceIsNotZero = self.m_DistanceIsNotZero if self.m_DistanceIsNotZero > 30 else (
                    self.m_DistanceIsNotZero + 1 if abs(tDistance - 0) > 1e-8 else self.m_DistanceIsNotZero)
                if tDistance <= 0:
                    continue

                tfAngle_H = float((data[i] << 8) + data[i + 1])
                if tfAngle_H > 32767:
                    tfAngle_H -= 65536
                tfAngle_H *= self.m_fH_AngleAcc

                tTempAngle = data[i + 2]
                tChannelID = tTempAngle >> 6
                tSymmbol = (tTempAngle >> 5) & 0x01
                if tSymmbol == 1:
                    iAngle_V = (data[i + 2] << 8) + data[i + 3]
                    tfAngle_V = float(iAngle_V | 0xc000)
                    if tfAngle_V > 32767:
                        tfAngle_V -= 65536
                else:
                    iAngle_Hight = tTempAngle & 0x3f
                    tfAngle_V = float((iAngle_Hight << 8) + data[i + 3])
                tfAngle_V *= self.m_fV_AngleAcc

                tDistance *= self.m_fDistanceAcc
                intensity = data[i + 7]

                tDistance_2 = ((data[i + 8] << 16) + (data[i + 9] << 8) + data[i + 10]) * self.m_fDistanceAcc
                intensity_2 = data[i + 11]

                if tChannelID >= len(self.mLidaFilterParamDisplayValue['mChannelVector']):
                    logger.error("id无效")
                    continue
                if self.mLidaFilterParamDisplayValue['mChannelVector'][tChannelID] == 0:
                    logger.error("通道未启用")
                    continue
                if not self.mLidaFilterParamDisplayValue['mMin_Distance'] <= tDistance <= \
                       self.mLidaFilterParamDisplayValue['mMax_Distance']:
                    logger.error("距离超出范围")
                    continue
                if not self.mLidaFilterParamDisplayValue['mMin_Intensity'] <= intensity <= \
                       self.mLidaFilterParamDisplayValue['mMax_Intensity']:
                    logger.error("反射强度超出范围")
                    continue

                m_point = self.XYZ_calculate(tChannelID, tfAngle_H, tfAngle_V, tDistance, tDistance_2)

                if not self.mLidaFilterParamDisplayValue['mMin_HanleValue'] <= tfAngle_H <= \
                       self.mLidaFilterParamDisplayValue['mMax_HanleValue']:
                    logger.error("水平角度超出范围")
                    continue
                if not self.mLidaFilterParamDisplayValue['mMin_VanleValue'] <= tfAngle_V <= \
                       self.mLidaFilterParamDisplayValue['mMax_VanleValue']:
                    logger.error("垂直角度超出范围")
                    continue

                m_DataT = LidarData(
                    ID=tChannelID,
                    H_angle=tfAngle_H,
                    V_angle=tfAngle_V,
                    Distance=tDistance,
                    Intensity=intensity,
                    X=m_point['x1'],
                    Y=m_point['y1'],
                    Z=m_point['z1'],
                    Mtimestamp_nsce=self.allTimestamp - self.PointIntervalT_ns * (98 - i // 12)
                )
                self.LidarPerFrameDatePrt_Get.append(m_DataT)

                m_DataT2 = LidarData(
                    ID=tChannelID,
                    H_angle=tfAngle_H,
                    V_angle=tfAngle_V,
                    Distance=tDistance_2,
                    Intensity=intensity_2,
                    X=m_point['x2'],
                    Y=m_point['y2'],
                    Z=m_point['z2'],
                    Mtimestamp_nsce=self.allTimestamp - self.PointIntervalT_ns * (98 - i // 12)
                )
                self.LidarPerFrameDatePrt_Get.append(m_DataT2)

    def XYZ_calculate(self, tChannelID, fAngle_H, fAngle_V, tDistance, tDistance2):
        if self.lidar_EchoModel in [0x11, 0x12]:
            point = {
                'x1': tDistance * self.cosAngleValue[negative2positive(fAngle_V)] * self.sinAngleValue[
                    negative2positive(fAngle_H)],
                'y1': tDistance * self.cosAngleValue[negative2positive(fAngle_V)] * self.cosAngleValue[
                    negative2positive(fAngle_H)],
                'z1': tDistance * self.sinAngleValue[negative2positive(fAngle_V)],
                'x2': 0,
                'y2': 0,
                'z2': 0
            }
            if tDistance2 > 0:
                point['x2'] = tDistance2 * self.cosAngleValue[negative2positive(fAngle_V)] * self.sinAngleValue[
                    negative2positive(fAngle_H)]
                point['y2'] = tDistance2 * self.cosAngleValue[negative2positive(fAngle_V)] * self.cosAngleValue[
                    negative2positive(fAngle_H)]
                point['z2'] = tDistance2 * self.sinAngleValue[negative2positive(fAngle_V)]
        else:
            point = {
                'x1': 0,
                'y1': 0,
                'z1': 0,
                'x2': 0,
                'y2': 0,
                'z2': 0
            }
            fPutTheMirrorOffAngle = float(self.m_fPutTheMirrorOffAngle[tChannelID])
            fGalvanometrtAngle = float(fAngle_V + self.m_DeadZoneOffset)

            fAngle_R0 = float(self.cosAngleValue[negative2positive(30)] * self.cosAngleValue[
                negative2positive(fPutTheMirrorOffAngle)] * self.cosAngleValue[
                                  negative2positive(fGalvanometrtAngle)] - \
                              self.sinAngleValue[negative2positive(fGalvanometrtAngle)] * self.sinAngleValue[
                                  negative2positive(fPutTheMirrorOffAngle)])

            fSinV_angle = float(2 * fAngle_R0 * self.sinAngleValue[negative2positive(fGalvanometrtAngle)] + \
                                self.sinAngleValue[negative2positive(fPutTheMirrorOffAngle)])
            fCosV_angle = float(np.sqrt(1 - fSinV_angle * fSinV_angle))

            fSinCite = float((2 * fAngle_R0 * self.cosAngleValue[negative2positive(fGalvanometrtAngle)] *
                              self.sinAngleValue[negative2positive(30)] - \
                              self.cosAngleValue[negative2positive(fPutTheMirrorOffAngle)] * self.sinAngleValue[
                                  negative2positive(60)]) / fCosV_angle)
            fCosCite = float(np.sqrt(1 - fSinCite * fSinCite))

            fSinCite_H = float(self.sinAngleValue[negative2positive(fAngle_H)] * fCosCite + self.cosAngleValue[
                negative2positive(fAngle_H)] * fSinCite)
            fCosCite_H = float(self.cosAngleValue[negative2positive(fAngle_H)] * fCosCite - self.sinAngleValue[
                negative2positive(fAngle_H)] * fSinCite)

            fAngle_H = np.degrees(np.arcsin(fSinCite_H))
            fAngle_V = np.degrees(np.arcsin(fSinV_angle))

            point['x1'] = float(tDistance * fCosV_angle * fSinCite_H)
            point['y1'] = float(tDistance * fCosV_angle * fCosCite_H)
            point['z1'] = float(tDistance * fSinV_angle)

            if tDistance2 > 0:
                point['x2'] = float(tDistance2 * fCosV_angle * fSinCite_H)
                point['y2'] = float(tDistance2 * fCosV_angle * fCosCite_H)
                point['z2'] = float(tDistance2 * fSinV_angle)
        return point

    def send_lidar_data(self, save_path="./out"):
        logger.debug("正在写入雷达信息")

        # 检查距离数据是否有效
        if self.m_DistanceIsNotZero < self.m_DistanceThreshold:
            logger.error("数据错误！！！所有雷达距离值均为 0！！！")
            return

        # 重置距离计数器
        self.m_DistanceIsNotZero = 0

        with self.m_mutex:
            # 保存当前帧的点云数据
            print(len(self.LidarPerFrameDatePrt_Get))
            self.LidarPerFrameDatePer = self.LidarPerFrameDatePrt_Get.copy()
            # 重置点云数据缓冲区，准备接收下一帧数据
            self.LidarPerFrameDatePrt_Get.clear()

        # 将点云数据转换为 Open3D 格式并保存
        points = np.array([[data.X, data.Y, data.Z] for data in self.LidarPerFrameDatePer])
        intensity = np.array([data.Intensity] for data in self.LidarPerFrameDatePer)
        logger.info(f"intensity:{intensity}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # # 使用雷达时间戳
        # timestamp_nsce = self.allTimestamp  # 假设 self.allTimestamp 是雷达的时间戳（纳秒级）
        # timestamp_ms = timestamp_nsce // 1000000  # 转换为毫秒级
        # 使用系统时间戳
        current_time = time.time()  # 当前时间（秒级时间戳）
        timestamp_nsce = int(current_time * 1e9)  # 转换为纳秒级时间戳
        timestamp_ms = timestamp_nsce // 1000000  # 转换为毫秒级

        # 转换为年月日时分秒格式
        dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
        timestamp_str = dt.strftime("%Y_%m_%d_%H_%M_%S")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = f"{save_path}/pointcloud_{timestamp_str}.pcd"
        try:
            o3d.io.write_point_cloud(filename, pcd)
            logger.info(f"点云数据已保存为文件：{filename}")
        except Exception as e:
            logger.error(f"保存点云文件失败: {e}")

import socket
import time
import numpy as np

def generate_lidar_data(ip, port, num_frames=10):
    # 定义帧分割标志
    delimiter = b'\xff\xaa\xbb\xcc\xdd'

    # 创建 UDP 套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for frame_id in range(num_frames):
        # 生成以太网包头（42字节）
        ethernet_header = bytearray(42)
        for i in range(42):
            ethernet_header[i] = np.random.randint(0, 256)

        # 生成点云数据（1192字节，每8字节一个点）
        point_cloud_data = bytearray()
        for i in range(149):  # 1192 / 8 = 149 个点
            # 生成随机的水平和垂直角度
            h_angle = np.random.randint(-32768, 32767)  # 限制在 -32768 到 32767
            v_angle = np.random.randint(-32768, 32767)  # 限制在 -32768 到 32767

            # 生成随机的距离和强度
            distance = np.random.randint(0, 100000) / 1000.0
            intensity = np.random.randint(0, 255)

            # 将角度和距离转换为字节
            h_angle_bytes = h_angle.to_bytes(2, byteorder='big', signed=True)
            v_angle_bytes = v_angle.to_bytes(2, byteorder='big', signed=True)
            distance_bytes = int(distance * 1000).to_bytes(3, byteorder='big', signed=False)
            intensity_bytes = intensity.to_bytes(1, byteorder='big', signed=False)

            # 将数据添加到点云数据中
            point_cloud_data.extend(h_angle_bytes)
            point_cloud_data.extend(v_angle_bytes)
            point_cloud_data.extend(distance_bytes)
            point_cloud_data.extend(intensity_bytes)

        # 生成附加信息（14字节）
        packet_counter = frame_id.to_bytes(2, byteorder='big', signed=False)
        utc_time = int(time.time()).to_bytes(6, byteorder='big', signed=False)
        timestamp_ns = int(time.time_ns() % 1000000000).to_bytes(4, byteorder='big', signed=False)
        radar_model = b'\x01'
        echo_info = b'\x00'

        # 组装数据包
        data = ethernet_header + point_cloud_data + packet_counter + utc_time + timestamp_ns + radar_model + echo_info

        # 随机插入帧分割标志
        insert_positions = sorted(np.random.choice(len(data), size=5, replace=False))
        for pos in insert_positions:
            data[pos:pos] = delimiter

        # 发送数据
        sock.sendto(data, (ip, port))
        print(f"发送帧 {frame_id + 1}/{num_frames} 到 {ip}:{port}")
        time.sleep(0.1)  # 模拟数据发送间隔

    sock.close()


def lidar_data_processing():
    # 初始化激光雷达数据处理类
    lidar = GetLidarData_LS()
    # 启动激光雷达数据处理
    lidar.LidarRun()


# 主程序入口
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("程序启动，开始监听激光雷达数据")

    # 启动雷达数据处理线程
    lidar_thread = threading.Thread(target=lidar_data_processing)
    lidar_thread.start()

    # 启动雷达数据生成线程
    generate_thread = threading.Thread(target=generate_lidar_data, args=(HOST, PORT, 10))
    generate_thread.start()

    # 等待雷达数据生成线程结束
    generate_thread.join()

    # 停止雷达数据处理线程
    lidar_thread.join()
