import os
import sys
import socket
import logging
import time
import queue
import traceback
import datetime
from dataclasses import dataclass
from collections import deque
from multiprocessing import Process,Queue,Lock,Manager
from logging.handlers import RotatingFileHandler
import multiprocessing
import numpy as np
import open3d as o3d
import signal
from src.tools.RigidTransform import point_cloud_coord_transform


# sudo kill -9 $(lsof -t -i:2469)

# print("可用的网络接口:", [iface for iface in socket.if_nameindex()])

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

def negative2positive(value):
    value_t = int(round(value * 1000))
    return value_t % 360000 if value_t >=0 else (value_t % -360000) + 360000


class UdpListener(Process): # 孙
    def __init__(self,status_dict,config_dict, data_queue,name):
        super().__init__(name=name)
        self.current_process_pid = os.getpid()
        self.data_queue = data_queue
        self.config_dict = config_dict
        self.status_dict = status_dict
        self.ip = self.config_dict["network_interface_ip"]
        self.port = self.config_dict["network_interface_port"]
        self.status_dict[self.ip] = {
            "status":True,
            "pid":self.current_process_pid
        }
        # os.system(f"sudo kill -9 $(lsof -t -i:{self.port})")
        # 新加属性，帧分割标识
        self.buffer = bytearray()
        self.isWork = False

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(5)

        self.isWork = True
        while self.isWork:
            try:
                data, addr = self.sock.recvfrom(2048)
                lidar_ip = addr[0]
                if self.data_queue.qsize() > 2:
                    if data[0] in [0x00, 0xa5] and data[1] == 0xff and data[2] == 0x00 and data[3] == 0x5a:
                        self.data_queue.put([lidar_ip, data])
                    else:
                        continue
                else:
                    self.data_queue.put([lidar_ip, data])
                # print("udp",[addr,data])



            except socket.timeout:
                print(f"雷达监听进程{self.current_process_pid}异常:{self.ip}[{self.port}]接收超时，继续监听...")
            except KeyboardInterrupt:
                print(f"雷达监听进程{self.current_process_pid},监测到Ctrl+C")
        print("UdpListener Ended")


    def cleanup(self):
        try:
            if self.sock:
                print(f"[PID:{self.current_process_pid}] 关闭套接字")
                self.sock.close()
            print(f"[PID:{self.current_process_pid}] 进程退出")
        except:
            pass

    def stop_listen_process(self):
        try:
            self.isWork = False
            time.sleep(1)
            self.cleanup()
        except:
            pass
        self.terminate()
        self.join()



class MultiLS25E(Process):
    def __init__(self,config_dict, queue):
        super().__init__(name="ML_MultiLS25E")
        self.config_dict = config_dict
        self.get_points_cloud_queue = queue
        self.list_lidar_config = self.config_dict["network_interface_list"]
        self.lidar_info_list = self.config_dict["lidar_info_list"]
        self.lidar_NUMS = 1
        self.dict_ip_calibration_matrix = {}
        self.dict_ip_decoding_config ={}
        self.list_udp_listener_process = []
        self.dict_lidar_queue = {}
        self.manager = Manager()
        self.status_dict = self.manager.dict()
        self.udp_data_queue = Queue()
        self.defult_decoding_config = {
            "m_HorizontalPoints":0,
            "m_StackFrame": 0 if self.lidar_NUMS == 1 else self.lidar_NUMS,
            "m_fV_AngleAcc":0.0025,
            "m_fH_AngleAcc":0.01,
            "m_fDistanceAcc":0.001,
            "m_fPutTheMirrorOffAngle":np.array([1.5, -0.5, 0.5, -1.5]),
            "m_DeadZoneOffset": 6.37,
            "m_UTC_Time": None,
            "lastAllTimestamp":0,
            "lidar_EchoModel":0,
            "PointIntervalT_ns":0,
            "count":0,
            "m_messageCount":0,
            "allTimestamp":0,
            "m_DistanceIsNotZero":0,
            "m_DistanceThreshold":20,
            "LidarPerFrameDatePrt_Get":deque(),
            "PointCloudLastData": deque(),
            "tempPointCloud": deque(),
            "cosAngleValue": np.cos(np.arange(360000) / 1000.0 * np.pi / 180.0),
            "sinAngleValue":np.sin(np.arange(360000) / 1000.0 * np.pi / 180.0),
            "mLidaFilterParamDisplayValue":{
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
        }
        self.create_decoding_config()
        print("LS配置初始化完成")
        

    def clearQueue(self, q):
        q.clear()  # 使用 deque 的 clear 方法    

    def UDP_Decode(self,key_ip,data):
        if data[0] in [0x00, 0xa5] and data[1] == 0xff and data[2] == 0x00 and data[3] == 0x5a:
            self.dict_ip_decoding_config[key_ip]["m_HorizontalPoints"] = (data[184] << 8) + data[185]
            if data[231] in [64, 65]:
                self.dict_ip_decoding_config[key_ip]["m_StackFrame"] = 2
            elif data[231] == 192:
                self.dict_ip_decoding_config[key_ip]["m_StackFrame"] = 1
                self.dict_ip_decoding_config[key_ip]["m_fV_AngleAcc"] = 0.01
                self.dict_ip_decoding_config[key_ip]["fDistanceAcc"] = 0.004
                self.dict_ip_decoding_config[key_ip]["m_fPutTheMirrorOffAngle"] = np.array([1.5, 0.5, -0.5, -1.5])
                self.dict_ip_decoding_config[key_ip]["m_DeadZoneOffset"] = 10.82
            else:
                self.dict_ip_decoding_config[key_ip]["m_fDistanceAcc"] = 0.001
                self.dict_ip_decoding_config[key_ip]["m_fH_AngleAcc"] = 0.01
                self.dict_ip_decoding_config[key_ip]["m_fV_AngleAcc"] = 0.0025
                self.dict_ip_decoding_config[key_ip]["m_StackFrame"] = 1
                self.dict_ip_decoding_config[key_ip]["m_DeadZoneOffset"] = 6.37
                self.dict_ip_decoding_config[key_ip]["m_fPutTheMirrorOffAngle"] = np.array([1.5, -0.5, 0.5, -1.5])
            majorVersion = data[1202]
            minorVersion1 = data[1203] // 16
            minorVersion2 = data[1203] % 16
            if majorVersion > 1 or (majorVersion == 1 and minorVersion1 > 1):
                self.dict_ip_decoding_config[key_ip]["m_fV_AngleAcc"] = 0.0025
            else:
                self.dict_ip_decoding_config[key_ip]["m_fV_AngleAcc"] = 0.01
            return [False,"机器参数配置",[]]
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
            self.dict_ip_decoding_config[key_ip]["allTimestamp"] = timestamp_s * 1000000000 + timestamp_nsce
        else:
            self.dict_ip_decoding_config[key_ip]["m_UTC_Time"] = {
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
                    self.dict_ip_decoding_config[key_ip]["m_UTC_Time"]['year'],
                    self.dict_ip_decoding_config[key_ip]["m_UTC_Time"]['month'],
                    self.dict_ip_decoding_config[key_ip]["m_UTC_Time"]['day'],
                    self.dict_ip_decoding_config[key_ip]["m_UTC_Time"]['hour'], 
                    self.dict_ip_decoding_config[key_ip]["m_UTC_Time"]['minute'], 
                    self.dict_ip_decoding_config[key_ip]["m_UTC_Time"]['second']
                )
                timestamp_s = int(dt.timestamp())
                self.dict_ip_decoding_config[key_ip]["allTimestamp"] = timestamp_s * 1000000000 + timestamp_nsce
            except ValueError as e:
                print(f"时间数据错误: {e}")
                return [False,f"时间数据错误: {e}",[]]
        if data[1204] == 192 and self.dict_ip_decoding_config[key_ip]["m_HorizontalPoints"] != 0:
            self.dict_ip_decoding_config[key_ip]["m_StackFrame"] = 1
            self.dict_ip_decoding_config[key_ip]["m_fV_AngleAcc"] = 0.01
            self.dict_ip_decoding_config[key_ip]["m_fDistanceAcc"] = 0.004
            self.dict_ip_decoding_config[key_ip]["m_fPutTheMirrorOffAngle"] = np.array([1.5, 0.5, -0.5, -1.5])
            self.dict_ip_decoding_config[key_ip]["m_DeadZoneOffset"] = 10.82

        self.dict_ip_decoding_config[key_ip]["lidar_EchoModel"] = data[1205]

        if self.dict_ip_decoding_config[key_ip]["lidar_EchoModel"] in [0x02, 0x12]:
            self.handleDoubleEcho(key_ip,data)
        else:
            self.handleSingleEcho(key_ip,data)
        self.dict_ip_decoding_config[key_ip]["lastAllTimestamp"] = self.dict_ip_decoding_config[key_ip]["allTimestamp"]

    def handleSingleEcho(self,key_ip, data):
        self.dict_ip_decoding_config[key_ip]["PointIntervalT_ns"] = (self.dict_ip_decoding_config[key_ip]["allTimestamp"] - self.dict_ip_decoding_config[key_ip]["lastAllTimestamp"]) / 149.0
        for i in range(0, 1192, 8):
            if data[i] == 0xff and data[i + 1] == 0xaa and data[i + 2] == 0xbb and data[i + 3] == 0xcc and data[i + 4] == 0xdd:
                self.dict_ip_decoding_config[key_ip]["count"] += 1
            if self.dict_ip_decoding_config[key_ip]["count"] == self.lidar_NUMS:
                print(f"捕获到共{self.lidar_NUMS}帧信息")
                # print(f"m_StackFrame:{self.m_StackFrame}")
                if self.dict_ip_decoding_config[key_ip]["m_StackFrame"] > 1:
                    self.dict_ip_decoding_config[key_ip]["tempPointCloud"] = self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"].copy()
                    self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"] = self.dict_ip_decoding_config[key_ip]["PointCloudLastData"]  + self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"]
                    self.dict_ip_decoding_config[key_ip]["PointCloudLastData"]  = self.dict_ip_decoding_config[key_ip]["tempPointCloud"]
                self.send_lidar_data(key_ip)
                self.dict_ip_decoding_config[key_ip]["count"] = 0
                self.dict_ip_decoding_config[key_ip]["m_messageCount"] = 0
                continue
            if len(self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"]) > 3000000:
                if self.dict_ip_decoding_config[key_ip]["m_messageCount"] == 1:
                    continue
                print("Frame synchronization failure!!!")
                self.dict_ip_decoding_config[key_ip]["m_messageCount"] = 1
                continue
            if self.dict_ip_decoding_config[key_ip]["count"] == 0:
                tDistance = float((data[i + 4] << 16) + (data[i + 5] << 8) + data[i + 6])
                self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] = self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] if self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] > 30 else (
                    self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] + 1 if abs(tDistance - 0) > 1e-8 else self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"])
                if tDistance <= 0:
                    continue        
                tfAngle_H = float((data[i] << 8) + data[i + 1])
                if tfAngle_H > 32767:
                    tfAngle_H -= 65536
                tfAngle_H *= self.dict_ip_decoding_config[key_ip]["m_fH_AngleAcc"]
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
                tfAngle_V *= self.dict_ip_decoding_config[key_ip]["m_fV_AngleAcc"]

                tDistance *= self.dict_ip_decoding_config[key_ip]["m_fDistanceAcc"]
                intensity = data[i + 7]

                if tChannelID >= len(self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mChannelVector']):
                    print("id无效")
                    continue
                if self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mChannelVector'][tChannelID] == 0:
                    print("通道未启用")
                    continue

                if not self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMin_Distance'] <= tDistance <= \
                       self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMax_Distance']:
                    print(f"距离超出范围")
                    continue
                if not self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMin_Intensity'] <= intensity <= \
                       self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMax_Intensity']:
                    print("反射强度超出范围")
                    continue

                m_point = self.XYZ_calculate(key_ip, tChannelID, tfAngle_H, tfAngle_V, tDistance, -1)

                if not self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMin_HanleValue'] <= tfAngle_H <= \
                       self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMax_HanleValue']:
                    print("水平角度超出范围")
                    continue
                if not self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMin_VanleValue'] <= tfAngle_V <= \
                       self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMax_VanleValue']:
                    print("垂直角度超出范围")
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
                    Mtimestamp_nsce=self.dict_ip_decoding_config[key_ip]["allTimestamp"] - self.dict_ip_decoding_config[key_ip]["PointIntervalT_ns"] * (148 - i // 8)
                )
                self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"].append(m_DataT)

    def handleDoubleEcho(self,key_ip,data):
        self.dict_ip_decoding_config[key_ip]["PointIntervalT_ns"] = (self.dict_ip_decoding_config[key_ip]["allTimestamp"] - self.dict_ip_decoding_config[key_ip]["lastAllTimestamp"]) / 99
        for i in range(0, 1188, 12):
            if data[i] == 0xff and data[i + 1] == 0xaa and data[i + 2] == 0xbb and data[i + 3] == 0xcc and data[i + 4] == 0xdd:
                self.dict_ip_decoding_config[key_ip]["count"] += 1
            if self.dict_ip_decoding_config[key_ip]["count"] == self.lidar_NUMS:
                print(f"捕获到共{self.lidar_NUMS}帧信息")
                if self.dict_ip_decoding_config[key_ip]["m_StackFrame"] > 1:
                    self.dict_ip_decoding_config[key_ip]["tempPointCloud"] = self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"].copy()
                    self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"] = self.dict_ip_decoding_config[key_ip]["PointCloudLastData"] + self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"]
                    self.dict_ip_decoding_config[key_ip]["PointCloudLastData"] = self.dict_ip_decoding_config[key_ip]["tempPointCloud"]
                self.send_lidar_data(key_ip)
                self.dict_ip_decoding_config[key_ip]["count"] = 0
                self.dict_ip_decoding_config[key_ip]["m_messageCount"] = 0
                continue
            if len(self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"]) > 3000000:
                if self.dict_ip_decoding_config[key_ip]["m_messageCount"] == 1:
                    continue
                print("Frame synchronization failure!!!")
                self.dict_ip_decoding_config[key_ip]["m_messageCount"] = 1
                continue
            if self.dict_ip_decoding_config[key_ip]["count"] == 0:
                tDistance = (data[i + 4] << 16) + (data[i + 5] << 8) + data[i + 6]
                self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] = self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] if self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] > 30 else (
                    self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] + 1 if abs(tDistance - 0) > 1e-8 else self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"])
                if tDistance <= 0:
                    continue
                tfAngle_H = float((data[i] << 8) + data[i + 1])
                if tfAngle_H > 32767:
                    tfAngle_H -= 65536
                tfAngle_H *= self.dict_ip_decoding_config[key_ip]["m_fH_AngleAcc"]
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
                tfAngle_V *= self.dict_ip_decoding_config[key_ip]["m_fV_AngleAcc"]

                tDistance *= self.dict_ip_decoding_config[key_ip]["m_fDistanceAcc"]
                intensity = data[i + 7]             

                tDistance_2 = ((data[i + 8] << 16) + (data[i + 9] << 8) + data[i + 10]) * self.dict_ip_decoding_config[key_ip]["m_fDistanceAcc"]
                intensity_2 = data[i + 11]                   
                if tChannelID >= len(self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mChannelVector']):
                    print("id无效")
                    continue
                if self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mChannelVector'][tChannelID] == 0:
                    print("通道未启用")
                    continue
                if not self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMin_Distance'] <= tDistance <= \
                       self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMax_Distance']:
                    print("距离超出范围")
                    continue
                if not self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMin_Intensity'] <= intensity <= \
                       self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMax_Intensity']:
                    print("反射强度超出范围")
                    continue

                m_point = self.XYZ_calculate(key_ip, tChannelID, tfAngle_H, tfAngle_V, tDistance, tDistance_2)

                if not self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMin_HanleValue'] <= tfAngle_H <= \
                       self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMax_HanleValue']:
                    print("水平角度超出范围")
                    continue
                if not self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMin_VanleValue'] <= tfAngle_V <= \
                       self.dict_ip_decoding_config[key_ip]["mLidaFilterParamDisplayValue"]['mMax_VanleValue']:
                    print("垂直角度超出范围")
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
                    Mtimestamp_nsce=self.dict_ip_decoding_config[key_ip]["allTimestamp"] - self.dict_ip_decoding_config[key_ip]["PointIntervalT_ns"] * (98 - i // 12)
                )
                self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"].append(m_DataT)                

                m_DataT2 = LidarData(
                    ID=tChannelID,
                    H_angle=tfAngle_H,
                    V_angle=tfAngle_V,
                    Distance=tDistance_2,
                    Intensity=intensity_2,
                    X=m_point['x2'],
                    Y=m_point['y2'],
                    Z=m_point['z2'],
                    Mtimestamp_nsce=self.dict_ip_decoding_config[key_ip]["allTimestamp"] - self.dict_ip_decoding_config[key_ip]["PointIntervalT_ns"] * (98 - i // 12)
                )
                self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"].append(m_DataT2)

    def XYZ_calculate(self,key_ip, tChannelID, fAngle_H, fAngle_V, tDistance, tDistance2):
        if self.dict_ip_decoding_config[key_ip]["lidar_EchoModel"] in [0x11, 0x12]:
            point = {
                'x1': tDistance * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][negative2positive(fAngle_V)] * self.dict_ip_decoding_config[key_ip]["sinAngleValue"][
                    negative2positive(fAngle_H)],
                'y1': tDistance * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][negative2positive(fAngle_V)] * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][
                    negative2positive(fAngle_H)],
                'z1': tDistance * self.dict_ip_decoding_config[key_ip]["sinAngleValue"][negative2positive(fAngle_V)],
                'x2': 0,
                'y2': 0,
                'z2': 0
            }
            if tDistance2 > 0:
                point['x2'] = tDistance2 * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][negative2positive(fAngle_V)] * self.dict_ip_decoding_config[key_ip]["sinAngleValue"][
                    negative2positive(fAngle_H)]
                point['y2'] = tDistance2 * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][negative2positive(fAngle_V)] * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][
                    negative2positive(fAngle_H)]
                point['z2'] = tDistance2 * self.dict_ip_decoding_config[key_ip]["sinAngleValue"][negative2positive(fAngle_V)]
        else:
            point = {
                'x1': 0,
                'y1': 0,
                'z1': 0,
                'x2': 0,
                'y2': 0,
                'z2': 0
            }
            fPutTheMirrorOffAngle = float(self.dict_ip_decoding_config[key_ip]["m_fPutTheMirrorOffAngle"][tChannelID])
            fGalvanometrtAngle = float(fAngle_V + self.dict_ip_decoding_config[key_ip]["m_DeadZoneOffset"])

            fAngle_R0 = float(self.dict_ip_decoding_config[key_ip]["cosAngleValue"][negative2positive(30)] * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][
                negative2positive(fPutTheMirrorOffAngle)] * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][
                                  negative2positive(fGalvanometrtAngle)] - \
                              self.dict_ip_decoding_config[key_ip]["sinAngleValue"][negative2positive(fGalvanometrtAngle)] * self.dict_ip_decoding_config[key_ip]["sinAngleValue"][
                                  negative2positive(fPutTheMirrorOffAngle)])
            
            fSinV_angle = float(2 * fAngle_R0 * self.dict_ip_decoding_config[key_ip]["sinAngleValue"][negative2positive(fGalvanometrtAngle)] + \
                                self.dict_ip_decoding_config[key_ip]["sinAngleValue"][negative2positive(fPutTheMirrorOffAngle)])
            fCosV_angle = float(np.sqrt(1 - fSinV_angle * fSinV_angle))

            fSinCite = float((2 * fAngle_R0 * self.dict_ip_decoding_config[key_ip]["cosAngleValue"][negative2positive(fGalvanometrtAngle)] *
                              self.dict_ip_decoding_config[key_ip]["sinAngleValue"][negative2positive(30)] - \
                              self.dict_ip_decoding_config[key_ip]["cosAngleValue"][negative2positive(fPutTheMirrorOffAngle)] * self.dict_ip_decoding_config[key_ip]["sinAngleValue"][
                                  negative2positive(60)]) / fCosV_angle)
            fCosCite = float(np.sqrt(1 - fSinCite * fSinCite))

            fSinCite_H = float(self.dict_ip_decoding_config[key_ip]["sinAngleValue"][negative2positive(fAngle_H)] * fCosCite + self.dict_ip_decoding_config[key_ip]["cosAngleValue"][
                negative2positive(fAngle_H)] * fSinCite)
            fCosCite_H = float(self.dict_ip_decoding_config[key_ip]["cosAngleValue"][negative2positive(fAngle_H)] * fCosCite - self.dict_ip_decoding_config[key_ip]["sinAngleValue"][
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

    def send_lidar_data(self, key_ip):
        try:
            print("正在写入雷达信息")
            # 检查距离数据是否有效
            if self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] < self.dict_ip_decoding_config[key_ip]["m_DistanceThreshold"]:
                print("数据错误!!!所有雷达距离值均为 0!!!")
                return
            # 重置距离计数器
            self.dict_ip_decoding_config[key_ip]["m_DistanceIsNotZero"] = 0

            # 保存当前帧的点云数据
            print(len(self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"]))
            LidarPerFrameDatePer = self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"].copy()
            # 重置点云数据缓冲区，准备接收下一帧数据
            self.dict_ip_decoding_config[key_ip]["LidarPerFrameDatePrt_Get"].clear()

            # 将点云数据转换为 Open3D 格式并保存
            ori_xyzi = np.array([[data.X, data.Y, data.Z,data.Intensity] for data in LidarPerFrameDatePer])
            trans_xyzi = ori_xyzi.copy()
            xyz = trans_xyzi[:, :3]  # 取前3列，表示 x, y, z 坐标
            matrix = self.dict_ip_calibration_matrix[key_ip]
            trans_xyzi[:, :3] = point_cloud_coord_transform(xyz, is_inv=False, calibration_matrix=matrix)
            # 使用系统时间戳
            data = {
                "key_ip":key_ip,  
                "ori_xyzi":ori_xyzi.copy(),
                "trans_xyzi":trans_xyzi
            }
            self.get_points_cloud_queue.put(data)
        except:
            pass
        

    def create_decoding_config(self):
        for lidar_info in self.lidar_info_list:
            self.dict_ip_calibration_matrix[lidar_info["lidar_ip"]] = lidar_info["calibration_matrix"]
            self.dict_ip_decoding_config[lidar_info["lidar_ip"]] = self.defult_decoding_config
            
    def create_process(self):
        count = 0
        for lidar_config in self.list_lidar_config:
            udp_listener = UdpListener(self.status_dict,lidar_config,self.udp_data_queue,f"ML_UdpListener_{count}")
            self.list_udp_listener_process.append(udp_listener)
            count += 1

    def start_upd_listener_process(self):
        for udp_listener in self.list_udp_listener_process:
            udp_listener.start()
                
    def stop_udp_listener_process(self):
        for udp_listener in self.list_udp_listener_process:
            udp_listener.stop_listen_process()
        self.terminate()

    def run(self):
        try:
            print("开始udplisten")
            self.create_process()
            print("雷神udp创建成功")
            print(self.status_dict)
            print(len(self.list_udp_listener_process))
            self.start_upd_listener_process()
            print("LS Start")
            current_size = self.udp_data_queue.qsize()
            print(f"UDP Queue current_size：{current_size}")
            while True:
                # st_time = time.time()
                item = self.udp_data_queue.get(block=True)
                # print("-"*100)
                lidar_ip,data = item
                # print(addr[0])
                # print(data)
                del item
                # ----------------------------------------------------------------
                try:
                    self.UDP_Decode(key_ip=lidar_ip,data=data)
                except:
                    error_log = traceback.format_exc()
                    print(error_log)
                # print(f"UDP 读取耗时：{time.time() - st_time}s")
                # ----------------------------------------------------------------
        except KeyboardInterrupt:
            print("MultiL25E 检测到ctrl+c")
            # self.stop_udp_listener_process()
        except Exception:
            error_log = traceback.format_exc()
            print(error_log)

            # self.get_points_cloud_queue.put(item)



if __name__ == "__main__":
    config = {
        "lidar_info_list":[
            {"lidar_ip":"192.168.0.200","calibration_matrix":[[0.966323,-0.255392,-0.031539,0],[0.255392,0.936786,0.239181,0],[-0.031539, -0.239181, 0.970463,0],[0,0,0,1]]}
        ],
        "network_interface_list":[
            {
                "network_interface_ip":"192.168.0.2",
                "network_interface_port":2368,
            },
        ]
    }
    
    points_queue = Queue()
    App = MultiLS25E(config,points_queue)
    # 注册信号处理
    def signal_handler(sig, frame):
        """处理终止信号"""
        print(f"\n收到信号 {sig}，停止所有进程...")
        App.stop_udp_listener_process()
        sys.exit(0)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    print(f"signal.SIGTERM:{signal.SIGTERM},signal.SIGINT:{signal.SIGINT}")

    App.start()

    while True:
        item = points_queue.get(block=True)
        print(f"get {item}")









