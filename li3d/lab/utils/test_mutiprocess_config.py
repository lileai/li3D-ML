# test_config_dict_mp.py
import multiprocessing
import pickle
import sys
from pathlib import Path


class ConfigDict(dict):
    """
    纯标准库实现的「属性友好」字典，行为与 addict.Dict 保持一致：
    1. 支持 cfg.a.b.c 链式属性访问
    2. 缺失键抛 AttributeError（而非自动创建）
    3. 赋值时自动把嵌套 dict 递归转成 ConfigDict
    4. 无循环自引用，可被 pickle / multiprocessing 安全序列化
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化时把嵌套 dict 一次性转成 ConfigDict，保证后续链式访问
        for k, v in list(self.items()):
            if isinstance(v, dict):
                self[k] = ConfigDict(v)

    # ---------- 读属性 ----------
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from None

    # ---------- 写属性 ----------
    def __setattr__(self, name, value):
        # 如果赋的是 dict，继续递归包成 ConfigDict，保持类型一致性
        if isinstance(value, dict):
            value = ConfigDict(value)
        self[name] = value

    # ---------- 缺失键抛 KeyError（与原来一致） ----------
    def __missing__(self, key):
        raise KeyError(key)

    # ---------- 以下两个魔术方法显式声明状态，彻底避免循环引用 ----------
    def __getstate__(self):
        # 只导出纯字典数据，不含自引用
        return dict(self)

    def __setstate__(self, state):
        # 反序列化后把数据写回，同样递归转换嵌套 dict
        self.clear()
        self.update(ConfigDict(state))  # 用构造器保证嵌套一致性


# ---------------- 模拟一份嵌套配置 ----------------
def build_cfg():
    cfg = ConfigDict({
        'model': {
            'backbone': {'type': 'ResNet', 'depth': 50},
            'neck': {'type': 'FPN'}
        },
        'data': {'workers': 4}
    })
    # 链式赋值再嵌套一层
    cfg.model.head = {'type': 'FCNHead', 'num_classes': 19}
    return cfg


# ---------------- 子进程入口：验证反序列化 + 属性访问 ----------------
def worker(pickle_bytes):
    cfg = pickle.loads(pickle_bytes)  # 反序列化
    # 属性链式访问
    print(f"[Worker] backbone.type = {cfg.model.backbone.type}")
    print(f"[Worker] backbone.depth = {cfg.model.backbone.depth}")
    print(f"[Worker] head.num_classes = {cfg.model.head.num_classes}")
    # 返回结果给主进程（证明双向序列化 OK）
    return pickle.dumps(cfg)


# ---------------- 主进程：序列化 → 子进程 → 回收结果 ----------------
def main():
    multiprocessing.set_start_method('spawn', force=True)  # 强制 spawn
    cfg = build_cfg()

    # ① 主进程第一次序列化
    blob = pickle.dumps(cfg)
    print("[Main] pickle.dumps OK")

    # ② 启动子进程
    with multiprocessing.Pool(processes=1) as pool:
        ret_blob = pool.apply(worker, (blob,))

    # ③ 主进程再次反序列化子进程返回的对象
    ret_cfg = pickle.loads(ret_blob)
    print("[Main] worker returned:")
    print(ret_cfg)

    print("[Main] All done —— ConfigDict works fine under multiprocessing!")


if __name__ == '__main__':
    main()
