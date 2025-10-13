#!/usr/bin/env python3
# ----------------- 多雷达版本 -----------------
# 其余 import 与之前完全一致
import os
import cv2
import socket
import struct
import threading
import time
import logging
import crcmod
import ipaddress
import datetime
import numpy as np
import open3d as o3d
from queue import Queue, Empty
from collections import deque
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# ---------- 常量 ----------
BROADCAST_PORT = 55000
DEVICE_CMD_PORT = 65000     # 设备固定源端口
HEARTBEAT_INTERVAL = 1.0
HEARTBEAT_TIMEOUT  = 3.0
BASE_CMD_PORT = 50000       # 50000, 50010, 50020...
MAX_RADAR     = 10           # 最多同时支持 10 台

# ---------- 工具 ----------
crc16 = crcmod.mkCrcFun(0x11021, rev=True, initCrc=0x4C49)
crc32 = crcmod.mkCrcFun(0x104C11DB7, rev=True, initCrc=0x564F580A, xorOut=0xFFFFFFFF)

# 设置cmd_frame_data
def _set_cmd_data(cmd_frame, cmd_id, length, param, name):
    try:
        head = struct.pack(cmd_frame[0], *[0xaa, 0x01, length, 0x00, 0x00])
        crc_16 = struct.pack(cmd_frame[1], crc16(head))
        if type(param) == list:
            data = struct.pack(cmd_frame[2], *cmd_id, *param)
        elif param is None:
            data = struct.pack(cmd_frame[2], *cmd_id)
        else:
            data = struct.pack(cmd_frame[2], *cmd_id, param)

        crc_32 = struct.pack(cmd_frame[3], crc32(head + crc_16 + data))

        cmd_frame_data = head + crc_16 + data + crc_32
        return cmd_frame_data
    except:
        print(f'打包{name}命令时出错')

def _parseResp(binData):
    """返回 (good, cmd_type_str, cmd_set_str, cmd_id_str, payload_bytes)"""
    if len(binData) < 12:  # 最小帧长
        return False, '', '', '', b''

    # taipu_main_side. 头字段（小端）
    sof, ver, length, cmd_type, seq, hcrc = struct.unpack('<BBH B HH', binData[:9])
    if sof != 0xAA or ver != 1 or length > 1400 or length > len(binData):
        return False, '', '', '', b''

    # 2. 头 CRC
    if crc16(binData[:7]) & 0xFFFF != hcrc:
        return False, '', '', '', b''

    # 3. 整帧 CRC
    if crc32(binData[:-4]) & 0xFFFFFFFF != struct.unpack('<I', binData[-4:])[0]:
        return False, '', '', '', b''

    # 4. CMD_SET / CMD_ID
    cmd_set, cmd_id = struct.unpack_from('<BB', binData, 9)
    payload = binData[11:-4]  # 去掉包头+CRC

    # 5. 文字描述
    cmd_type_str = {0: 'CMD', 1: 'ACK', 2: 'MSG'}.get(cmd_type, '')
    cmd_set_str = {0: 'General', 1: 'Lidar', 2: 'Hub'}.get(cmd_set, '')
    cmd_id_str = str(cmd_id)

    return True, cmd_type_str, cmd_set_str, cmd_id_str, payload

def get_ack_and_parse(sock, length=1024, name=None, verse=True):
    try:
        data_ack, addr_ack = sock.recvfrom(length)
        if len(data_ack) < 12:  # 最短帧
            if verse:
                logging.warning(f'{name} ACK too short: {len(data_ack)}')
            return (False, '', '', '', b'')

        ack_data = _parseResp(data_ack)
        if verse:
            logging.info(f'{name} ACK parsed: {ack_data}')
        return ack_data
    except (socket.timeout, struct.error) as e:
        logging.warning(f'{name} ACK error: {e}')
        return (False, '', '', '', b'')

def get_ip_for_target(target_ip: str) -> str:
    """
    根据目标 IP 返回本机同一网段网卡的 IP
    例如 192.168.10.100 → 192.168.10.xxx
    """
    target = ipaddress.ip_address(target_ip)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # 连接目标设备，OS 会自动选择正确的网卡
        s.connect((target_ip, 55000))
        local_ip = s.getsockname()[0]
    return local_ip

# ---------- 单雷达 FSM（内部类） ----------
class SingleLivoxFSM(threading.Thread):
    def __init__(self, broadcast_code:str, device_addr, base_port:int, result_q:Queue):
        super().__init__(daemon=True)
        self.bc = broadcast_code
        self.device_addr = device_addr
        self.cmd_port   = base_port
        self.data_port  = base_port + 1
        self.imu_port   = base_port + 1  # 节省资源和点云data同一个port
        self.result_q   = result_q
        self.running    = True
        self.sock_cmd   = None
        self.user_ip    = None

    # -------- 与单雷达版本基本一致，仅端口不同 --------
    def run(self):
        self._switch_local_port()
        self._send_connect_request()
        self._start_heartbeat()

    def _switch_local_port(self):
        self.user_ip = get_ip_for_target(self.device_addr[0])
        self.sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_cmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_cmd.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock_cmd.bind((self.user_ip, self.cmd_port))
        logging.info('[%s] bind cmd=%d data=%d', self.bc, self.cmd_port, self.data_port)

    def _send_connect_request(self):
        for attempt in range(10):  # 最多重试 10 次
            param = [int(x) for x in self.user_ip.split('.')] + [self.data_port, self.cmd_port, self.imu_port]
            frame = _set_cmd_data(
                cmd_frame=['<BBHBH', '<H', '<BB4BHHH', '<I'],
                cmd_id=(0x00, 0x01), length=25, param=param, name=f'{self.bc}-handshake')
            self.sock_cmd.sendto(frame, (self.device_addr[0], DEVICE_CMD_PORT))
            ack = get_ack_and_parse(self.sock_cmd, name=f'{self.bc}-handshake')
            if ack[0]:
                self._set_sample_mode()
                self._start_sampling()
                return
            else:
                logging.warning('[%s] handshake failed (attempt %d)', self.bc, attempt + 1)
                time.sleep(1)
        # 最终失败
        logging.error('[%s] handshake failed after retries', self.bc)
        self.q.put((self.bc, False, self.user_ip, self.data_port))

    def _set_sample_mode(self):
        frame = _set_cmd_data(
            cmd_frame=['<BBHBH', '<H', '<BBB', '<I'],
            cmd_id=(0x01, 0x06), length=16, param=[0x03], name=f'{self.bc}-start')
        self.sock_cmd.sendto(frame, (self.device_addr[0], DEVICE_CMD_PORT))
        ack = get_ack_and_parse(self.sock_cmd, name=f'{self.bc}-start')
        ok = ack[0]
        self.result_q.put((self.bc, ok, self.user_ip, self.data_port))
        if ok:
            logging.info('[%s] set double back mode ok', self.bc)
        else:
            logging.warning('[%s] set double back mode failed', self.bc)


    def _start_sampling(self):
        frame = _set_cmd_data(
            cmd_frame=['<BBHBH', '<H', '<BBB', '<I'],
            cmd_id=(0x00, 0x04), length=16, param=[0x01], name=f'{self.bc}-start')
        self.sock_cmd.sendto(frame, (self.device_addr[0], DEVICE_CMD_PORT))
        ack = get_ack_and_parse(self.sock_cmd, name=f'{self.bc}-start')
        ok = ack[0]
        self.result_q.put((self.bc, ok, self.user_ip, self.data_port))
        if ok:
            logging.info('[%s] start sampling ok', self.bc)
        else:
            logging.warning('[%s] start sampling failed', self.bc)

    def _start_heartbeat(self):
        self.sock_cmd.settimeout(HEARTBEAT_TIMEOUT)
        while self.running:
            try:
                frame = _set_cmd_data(
                    cmd_frame=['<BBHBH', '<H', '<BB', '<I'],
                    cmd_id=(0x00, 0x03), length=15, param=[], name=f'{self.bc}-hb')
                self.sock_cmd.sendto(frame, (self.device_addr[0], DEVICE_CMD_PORT))
                ack = get_ack_and_parse(self.sock_cmd, name=f'{self.bc}-hb', verse=False)
                if ack[0]:
                    pass
                else:
                    logging.warning('[%s] heartbeat lost', self.bc)
            except Exception as e:
                logging.error('[%s] heartbeat exc: %s', self.bc, e)
            time.sleep(HEARTBEAT_INTERVAL)

    def stop(self):
        self.running = False
        if self.sock_cmd:
            self.sock_cmd.close()

# ---------- 雷达管理器 ----------
class LivoxManager(threading.Thread):
    def __init__(self, result_q:Queue):
        super().__init__(daemon=True)
        self.result_q = result_q
        self.running  = True
        self.seen = set()          # 已处理的广播码
        self.children = []         # SingleLivoxFSM 实例

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, taipu_main_side)
            sock.bind(('', BROADCAST_PORT))
            sock.settimeout(1.0)
            logging.info('Manager listening broadcast on %d', BROADCAST_PORT)
            port_idx = 0
            while self.running and port_idx < MAX_RADAR:
                try:
                    data, addr = sock.recvfrom(1024)
                except socket.timeout:
                    continue
                ok, cmd_type, cmd_set, cmd_id, payload = _parseResp(data)
                if not ok or cmd_type != 'MSG' or cmd_set != 'General' or cmd_id != '0':
                    continue
                if len(payload) < 11:
                    continue
                bc = payload[:11].decode('ascii', errors='ignore')
                if bc in self.seen:
                    continue
                self.seen.add(bc)
                logging.info('New radar %s from %s', bc, addr[0])
                child = SingleLivoxFSM(bc, addr, BASE_CMD_PORT + port_idx*10, self.result_q)  # 实例化一个雷达的状态机
                child.start()
                self.children.append(child)
                port_idx += 1
            logging.info('Manager finished, total %d radars', len(self.children))

    def stop(self):
        self.running = False
        for c in self.children:
            c.stop()

# ---------- 噪点过滤 ----------
def _filter_noise(pts):
    if pts.size == 0:
        return np.empty((0, 4), np.float32)
    tag = pts[:, 4].astype(np.uint8)
    mask = (tag & 0x0F) == 0
    return pts[mask, :4]

# ---------- 主函数 ----------
def parse_livox_datagram(data):
    if len(data) < 18:
        return np.empty((0, 4), np.float32), {}

    # 14 字节包头
    (version, slot_id, lidar_id, reserved,
     status_code, timestamp_type,
     data_type) = struct.unpack('<BBBBIBB', data[:10])
    timestamp = struct.unpack('<Q', data[10:18])[0]
    header = dict(version=version, slot_id=slot_id,
                  lidar_id=lidar_id, status_code=status_code,
                  timestamp_type=timestamp_type,
                  data_type=data_type,
                  timestamp_ns=timestamp)

    payload = data[18:]
    if data_type == 6:          # IMU
        return np.empty((0, 4), np.float32), header

    # 每点字节数（官方表）
    point_bytes = {2: 14, 3: 10, 4: 28, 5: 16, 7: 42, 8: 22}.get(data_type, 0)
    if point_bytes == 0 or len(payload) % point_bytes != 0:
        return np.empty((0, 4), np.float32), header

    buf = np.frombuffer(payload, dtype=np.uint8).reshape(-1, point_bytes)
    pts = []

    # ---------- 逐字节解码 ----------
    for p in buf:
        if data_type == 2:      # 单直角 14B
            x, y, z = struct.unpack('<iii', p[0:12])
            refl, tag = p[12], p[13]
            pts.append([x*1e-3, y*1e-3, z*1e-3, refl, tag])

        elif data_type == 3:    # 单球 10B
            d, zen, azi = struct.unpack('<IHH', p[0:8])
            refl, tag = p[8], p[9]
            zen, azi = np.deg2rad(zen*0.01), np.deg2rad(azi*0.01)
            d *= 1e-3
            x = d * np.sin(zen) * np.cos(azi)
            y = d * np.sin(zen) * np.sin(azi)
            z = d * np.cos(zen)
            pts.append([x, y, z, refl, tag])

        elif data_type == 4:    # 双直角 28B
            for off in (0, 14):
                x, y, z = struct.unpack('<iii', p[off:off+12])
                refl, tag = p[off+12], p[off+13]
                pts.append([x*1e-3, y*1e-3, z*1e-3, refl, tag])

        elif data_type == 5:    # 双球 16B
            zen, azi = struct.unpack('<HH', p[0:4])
            zen, azi = np.deg2rad(zen*0.01), np.deg2rad(azi*0.01)
            for off in (4, 10):
                d, refl, tag = struct.unpack('<IBB', p[off:off+6])
                d *= 1e-3
                x = d * np.sin(zen) * np.cos(azi)
                y = d * np.sin(zen) * np.sin(azi)
                z = d * np.cos(zen)
                pts.append([x, y, z, refl, tag])

        elif data_type == 7:    # 三直角 42B
            for off in (0, 14, 28):
                x, y, z = struct.unpack('<iii', p[off:off+12])
                refl, tag = p[off+12], p[off+13]
                pts.append([x*1e-3, y*1e-3, z*1e-3, refl, tag])

        elif data_type == 8:    # 三球 22B
            zen, azi = struct.unpack('<HH', p[0:4])
            zen, azi = np.deg2rad(zen*0.01), np.deg2rad(azi*0.01)
            for off in (4, 10, 16):
                d, refl, tag = struct.unpack('<IBB', p[off:off+6])
                d *= 1e-3
                x = d * np.sin(zen) * np.cos(azi)
                y = d * np.sin(zen) * np.sin(azi)
                z = d * np.cos(zen)
                pts.append([x, y, z, refl, tag])

    points = _filter_noise(np.array(pts, dtype=np.float32))
    return points, header

# ---------- demo ----------
if __name__ == '__main__':
    SAVE_DIR = r"../../data/pcd_files"
    os.makedirs(SAVE_DIR, exist_ok=True)

    q = Queue()
    mgr = LivoxManager(q)
    mgr.start()

    # 等待所有雷达握手+start_sampling完成
    radar_infos = []
    try:
        while True:
            try:
                bc, ok, ip, port = q.get(timeout=15)
                radar_infos.append((bc, ok, ip, port))
                logging.info('Radar %s result: ok=%s ip=%s port=%d', bc, ok, ip, port)
            except Empty:
                break
    except KeyboardInterrupt:
        pass

    if not radar_infos:
        print("❌ 未发现任何雷达"); mgr.stop(); exit(1)

    # 为每台雷达开独立线程收数
    def receiver(bc, ip, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        sock.settimeout(1.0)
        # buf 存 (pts, recv_ts)
        buf = deque(maxlen=10000)
        last_save = time.monotonic()
        logging.info('[%s] receiver started on %s:%d', bc, ip, port)

        try:
            while True:
                try:
                    data, _ = sock.recvfrom(2048)
                    recv_ts = time.time()  # 记录收包时刻
                except socket.timeout:
                    continue

                pts, hdr = parse_livox_datagram(data)
                if pts.shape[0] == 0:
                    continue
                buf.append((pts, recv_ts))

                # 0.5 s 写盘触发
                if time.monotonic() - last_save >= 0.5:
                    if buf:
                        # 1. 窗口右端
                        t_end = buf[-1][1]
                        # 2. 找第一个 ≥ t_end-0.5 的 tuple
                        left_idx = 0
                        while left_idx < len(buf) and t_end - buf[left_idx][1] > 0.5:
                            left_idx += 1
                        # 3. 扔掉左侧老数据
                        for _ in range(left_idx):
                            buf.popleft()
                        # 4. 拼窗口内点云
                        all_pts = np.concatenate([item[0] for item in buf], axis=0)

                        # 5. 按原流程保存
                        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        path = os.path.join(SAVE_DIR, f'{bc}_{ts}.pcd')
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(all_pts[:, :3])

                        intensity = all_pts[:, 3]
                        intensity_min, intensity_max = intensity.min(), intensity.max()
                        normalized_intensity = (
                                (intensity - intensity_min) / (intensity_max - intensity_min) * 255
                        ).astype(np.uint8)
                        colors = cv2.applyColorMap(normalized_intensity, cv2.COLORMAP_JET).reshape(-1, 3) / 255.0
                        pcd.colors = o3d.utility.Vector3dVector(colors)

                        o3d.io.write_point_cloud(path, pcd, print_progress=False)
                        logging.info('[%s] saved %d pts (%.2f s window) -> %s',
                                     bc, len(all_pts), 0.5, path)

                    last_save = time.monotonic()
        except KeyboardInterrupt:
            logging.info('[%s] user interrupt', bc)
        finally:
            sock.close()

    threads = []
    for bc, ok, ip, port in radar_infos:
        if ok:
            t = threading.Thread(target=receiver, args=(bc, ip, port), daemon=True)
            t.start()
            threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        mgr.stop()