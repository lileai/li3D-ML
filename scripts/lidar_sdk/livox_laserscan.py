# #!/usr/bin/env python3
import os
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

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s')

# ---------- 常量 ----------
BROADCAST_PORT = 55000
DEVICE_CMD_PORT = 65000  # 设备固定源端口
CMD_PORT = 50001
HEARTBEAT_INTERVAL = 1.0  # taipu_main_side Hz
HEARTBEAT_TIMEOUT = 3.0  # 3 秒无 ACK 即重连

# ---------- 工具 ----------

crc16 = crcmod.mkCrcFun(0x11021, rev=True, initCrc=0x4C49)
crc32 = crcmod.mkCrcFun(0x104C11DB7, rev=True,
                        initCrc=0x564F580A, xorOut=0xFFFFFFFF)


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


# ---------- 状态机 ----------
class LivoxFSM(threading.Thread):
    def __init__(self, result_queue: Queue):
        super().__init__(daemon=True)
        self._running = True
        self._state = 'MONITOR_55000'
        self._local_port = None
        self._sock_cmd = None
        self._device_addr = None
        self._last_ack_time = 0.0
        self._result_queue = result_queue  # ✅ 用来传值

    def run(self):
        self._monitor_55000()

    # ---------------- 55000 监听 ----------------
    def _monitor_55000(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', BROADCAST_PORT))
            logging.info('Monitor port %d', BROADCAST_PORT)

            while self._running:
                try:
                    data, addr = sock.recvfrom(1024)
                except OSError:
                    break

                # 使用统一的解析函数
                ok, cmd_type, cmd_set, cmd_id, payload = _parseResp(data)
                if not ok:
                    continue  # 校验失败，跳过

                # 只处理 MSG 类型广播
                if cmd_type != 'MSG':
                    continue

                # 只处理 CMD_SET=0x00, CMD_ID=0x00 的广播信息
                if cmd_set != 'General' or cmd_id != '0':
                    continue

                # 提取广播码（ASCII）
                if len(payload) < 11:
                    continue
                broadcast_code = payload[:11].decode('ascii', errors='ignore')
                self._device_addr = addr
                logging.info('Received broadcast: %s from %s', broadcast_code, addr[0])

                # 退出监听，进入下一步
                sock.close()
                self._switch_local_port()

    # ---------------- 切换本地端口 ----------------
    def _switch_local_port(self):
        # 建立指令端口套字
        self._local_port = CMD_PORT  # 50001
        self._sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_cmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock_cmd.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # 启用广播
        # 获取当前网卡的ip
        self.user_ip = get_ip_for_target(self._device_addr[0])
        logging.info(f"user_ip：{self.user_ip}")
        # 绑定指令端口
        self._sock_cmd.bind((self.user_ip, self._local_port))
        logging.info('Switch local port for cmd %d', self._local_port)
        # 进入发送握手指令状态
        self._send_connect_request()

    # ---------------- 连接请求 ----------------
    def _send_connect_request(self):
        """
        发送 Livox 网络握手请求（CMD_SET=0x00, CMD_ID=0x01）
        协议帧格式：
        +---------+---------+---------+---------+---------+---------+
        | SOF(taipu_main_side)  | VER(taipu_main_side)  | LEN(2)  | TYPE(taipu_main_side) | SEQ(2)  | HCRC(2) |
        +---------+---------+---------+---------+---------+---------+
        | CMD_SET(taipu_main_side) | CMD_ID(taipu_main_side) | PAYLOAD(25) | FCRC(4) |
        +----------------------------------------------------------+
        """
        param = [int(x) for x in self.user_ip.split('.')] + [self._local_port + 1,  # data_port: 50002
                                                             self._local_port,  # cmd_port: 50001
                                                             self._local_port + 1]  # imu_port: 50002
        cmd_frame = ['<BBHBH', '<H', '<BB4BHHH', '<I']
        cmd_id = (0x00, 0x01)
        length = 25
        name = "hand shake"
        full_frame = _set_cmd_data(cmd_frame=cmd_frame,
                                   cmd_id=cmd_id,
                                   length=length,
                                   param=param,
                                   name=name)
        # 3. 发送
        logging.info(self._device_addr)
        self._sock_cmd.sendto(full_frame, (self._device_addr[0], DEVICE_CMD_PORT))
        logging.info('Send connect request to %s:%d', self._device_addr[0], DEVICE_CMD_PORT)
        # 进入ack校验状态
        self._monitor_ack()

    # ---------------- 等待 ACK ----------------
    def _monitor_ack(self):
        self._sock_cmd.settimeout(HEARTBEAT_TIMEOUT)
        self._state = 'MONITOR_ACK'
        ack_data = get_ack_and_parse(self._sock_cmd, name="shake hand")
        if ack_data[0]:
            # 先尝试启动采样
            msg = self._start_sampling()
            threading.Thread(target=self._start_heartbeat,
                             daemon=True, name='heartbeat').start()
            return msg
        else:
            self._send_connect_request()

    # ---------------- 心跳 ----------------
    def _start_heartbeat(self):
        self._state = 'HEARTBEAT'
        self._last_ack_time = time.time()
        self._sock_cmd.settimeout(HEARTBEAT_TIMEOUT)
        try:
            while self._running:
                # taipu_main_side. 发送心跳
                frame = _set_cmd_data(
                    cmd_frame=['<BBHBH', '<H', '<BB', '<I'],
                    cmd_id=(0x00, 0x03),
                    length=15,
                    param=[],
                    name="heartbeat"
                )
                self._sock_cmd.sendto(frame, (self._device_addr[0], DEVICE_CMD_PORT))

                # 2. 用统一函数收 + 解析
                ack_data = get_ack_and_parse(self._sock_cmd, name="heartbeat")
                if ack_data[0]:
                    self._last_ack_time = time.time()
                    logging.info('Heartbeat ACK')
                else:
                    # 超时计数，决定是否重连
                    if time.time() - self._last_ack_time > HEARTBEAT_TIMEOUT:
                        logging.warning('Heartbeat timeout → reconnect')
                        self._switch_local_port()
                        break

                time.sleep(HEARTBEAT_INTERVAL)
        finally:
            sock.close()

    def _start_sampling(self):
        frame = _set_cmd_data(
            cmd_frame=['<BBHBH', '<H', '<BBB', '<I'],  # ✅ 注意这里是 <BBB，因为你 param=[0x01]
            cmd_id=(0x00, 0x04),
            length=16,
            param=[0x01],  # ✅ 开始采样
            name="start_sampling"
        )
        self._sock_cmd.sendto(frame, (self._device_addr[0], DEVICE_CMD_PORT))
        logging.info('Send start sampling command')

        ack_data = get_ack_and_parse(self._sock_cmd, name="start sampling")
        result = (ack_data[0], self.user_ip, self._local_port + 1)

        # ✅ 把结果放进队列
        self._result_queue.put(result)

        # ✅ 原逻辑保留，返回给内部调用者
        return result

    def stop(self):
        self._running = False
        if self._sock_cmd:
            self._sock_cmd.close()

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
            pts.append([x*1e-3, y*1e-3, z*1e-3, refl/255.0, tag])

        elif data_type == 3:    # 单球 10B
            d, zen, azi = struct.unpack('<IHH', p[0:8])
            refl, tag = p[8], p[9]
            zen, azi = np.deg2rad(zen*0.01), np.deg2rad(azi*0.01)
            d *= 1e-3
            x = d * np.sin(zen) * np.cos(azi)
            y = d * np.sin(zen) * np.sin(azi)
            z = d * np.cos(zen)
            pts.append([x, y, z, refl/255.0, tag])

        elif data_type == 4:    # 双直角 28B
            for off in (0, 14):
                x, y, z = struct.unpack('<iii', p[off:off+12])
                refl, tag = p[off+12], p[off+13]
                pts.append([x*1e-3, y*1e-3, z*1e-3, refl/255.0, tag])

        elif data_type == 5:    # 双球 16B
            zen, azi = struct.unpack('<HH', p[0:4])
            zen, azi = np.deg2rad(zen*0.01), np.deg2rad(azi*0.01)
            for off in (4, 10):
                d, refl, tag = struct.unpack('<IBB', p[off:off+6])
                d *= 1e-3
                x = d * np.sin(zen) * np.cos(azi)
                y = d * np.sin(zen) * np.sin(azi)
                z = d * np.cos(zen)
                pts.append([x, y, z, refl/255.0, tag])

        elif data_type == 7:    # 三直角 42B
            for off in (0, 14, 28):
                x, y, z = struct.unpack('<iii', p[off:off+12])
                refl, tag = p[off+12], p[off+13]
                pts.append([x*1e-3, y*1e-3, z*1e-3, refl/255.0, tag])

        elif data_type == 8:    # 三球 22B
            zen, azi = struct.unpack('<HH', p[0:4])
            zen, azi = np.deg2rad(zen*0.01), np.deg2rad(azi*0.01)
            for off in (4, 10, 16):
                d, refl, tag = struct.unpack('<IBB', p[off:off+6])
                d *= 1e-3
                x = d * np.sin(zen) * np.cos(azi)
                y = d * np.sin(zen) * np.sin(azi)
                z = d * np.cos(zen)
                pts.append([x, y, z, refl/255.0, tag])

    points = _filter_noise(np.array(pts, dtype=np.float32))
    return points, header


# ---------- demo ----------
if __name__ == '__main__':
    SAVE_DIR = "/home/hello/pcd_files"
    os.makedirs(SAVE_DIR, exist_ok=True)

    q = Queue()
    fsm = LivoxFSM(q)
    fsm.start()

    try:
        ok, ip, port = q.get(timeout=10)
        if not ok:
            print("❌ 采样启动失败")
            fsm.stop(); exit(1)

        print(f"✅ 采样成功，准备收包 IP={ip}, 端口={port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        sock.settimeout(1.0)

        frame_buffer = deque(maxlen=1000)    # 只保留最近 1000 帧
        last_save = time.monotonic()
        frame_cnt = 0

        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                logging.info("数据获取超时...")
                continue

            pts, hdr = parse_livox_datagram(data)
            if pts.shape[0] == 0:
                continue

            frame_buffer.append(pts)        # 存整帧
            frame_cnt += 1

            # ---- 每 0.5 秒保存一次 ----
            now = time.monotonic()
            if now - last_save >= 0.5:
                if frame_buffer:
                    all_pts = np.concatenate(list(frame_buffer), axis=0)
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    file_path = os.path.join(SAVE_DIR, f'livox_{ts}.pcd')

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(all_pts[:, :3])
                    colors = np.tile(all_pts[:, 3].reshape(-1, 1), (1, 3))
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    o3d.io.write_point_cloud(file_path, pcd, print_progress=False)
                    print(f"已保存 {file_path} 共 {len(all_pts)} 点 来自 {len(frame_buffer)} 帧")

                    frame_buffer.clear()  # 清空，开始下一轮 1000 帧
                last_save = now

    except KeyboardInterrupt:
        print("用户中断")
    finally:
        fsm.stop()
        sock.close()