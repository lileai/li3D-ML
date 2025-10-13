#coding=UTF-8 
import time
import threading
import cv2
import grpc
import numpy as np
# from log.log import log
import msg_pb2, msg_pb2_grpc


_HOST = '192.168.101.124'   #设置grpc服务ip
_PORT = '7908'        #设置grpc服务端口

conn = grpc.insecure_channel(_HOST + ':' + _PORT, options=(('grpc.enable_http_proxy', 0),('grpc.max_send_message_length', 10 * 4000 * 3000)))  # 监听频道
client = msg_pb2_grpc.FormatDataStub(channel=conn)  # 客户端使用Stub类发送请求,参数为频道,为了绑定链接

def run(mode,img):
    '''
    客户端练级服务端测试用的脚本
    :return: 无返回，但是会print服务端的输出
    '''

    img_shape = list(img.shape)
    img = bytes(img)
    response = client.get_roi_from_bytes(
        msg_pb2.request_bytes(
            mode=str(mode), image=img, image_shape=img_shape, confidence=0
        )
    )

    return response



if __name__ == '__main__':
    time1 = time.time()
    path = r'D:\program\li3D-ML\data\Qiaolin\video\images1\frame_00030.jpg'   #被测试的图片路径
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    time2 = time.time()
    # log.logger.info('读取图片耗时：{}'.format(time2-time1))
    mode = 1     #测试grpc服务的模式，例  1 或 xgq
    response = run(mode,img)
    # print(len(response.data))
    for i in range(len(response.data)):   #打印结果1
        print('检测框为：{}'.format(response.data[i].ROI))
        print('置信度为：{}'.format(response.data[i].confidence))
        print('类别：{}'.format(response.data[i].type))
        


