from multiprocessing import Process
import traceback
import os
import time


class AisFusionProcess(Process):
    def __init__(self, status_dict,config_dict,queue1,dict_AisInfo):
        super().__init__()
        self.status_dict = status_dict
        self.config_dict = config_dict
        self.queue1 = queue1 # 输入：UUID、box等信息
        self.dict_AisInfo = dict_AisInfo # 输出

    def data_acquisition(self):
        """
        从队列里面获取数据，并解包
        """
        pass

    def update_aisinfo(self):
        """
        获取Redis信息并更新AIS信息
        """
        print("123")
        

    def fusion(self):
        "激光雷达位置信息与GPS经纬度信息融合"
        pass

    def data_process(self):
        while True:
            try:
                pass
            except:
                pass

    def run(self):
        self.status_dict['AisFusionProcess_pid'] = os.getpid()
        self.status_dict['AisFusionProcess_status'] = 'running'
        try: 
            while True:
                # 模拟一些工作
                self.data_process()
                
        except Exception as e:
            self.status_dict['AisFusionProcess_status'] = 'error'
        finally:
            self.status_dict['AisFusionProcess_status'] = 'stopped'

if __name__ == "__main__":
    from multiprocessing import Queue, Manager
    class MyTest:
        def __init__(self,config):
            self.config = config
            self.queue_0 = Queue()
            self.manager = Manager()
            self.status_dict = self.manager.dict()
            self.config_dict = self.manager.dict(self.config)

            self.point_cloud_reading_process = AisFusionProcess(self.status_dict,self.config_dict,self.queue_0,self.queue_1,self.queue_4) # 点云获取，融合，写入队列1{time,fusion_points}

        def start_processes(self):
            """ 启动所有子进程 """
            self.point_cloud_reading_process.start()



        def terminate_processes(self):
            """ 终止所有进程 """
            # logging.info("Shutting down...")
            self.point_cloud_reading_process.terminate()

            # 等待进程退出
            self.point_cloud_reading_process.join()


        def run(self):
            """ 启动进程并监听队列 """
            try:
                self.start_processes()
                while True:
                    current_time = time.time()
                    item = self.queue_0.get(block=True)
                    xyzi = item["xyzi"]
            except KeyboardInterrupt:
                # 捕获键盘中断，优雅退出
                self.terminate_processes()
            except:
                error_log = traceback.format_exc()
                print(error_log)