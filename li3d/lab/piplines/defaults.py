"""
Default training/testing logic

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import argparse
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


from ..utils import comm
from ..utils.env import get_random_seed, set_seed
from ..utils.config import Config, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = None if seed is None else num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="D:\program\li3D-ML\li3d\configs\customer\customer.py", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser


def default_config_parser(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    # 1. 拿到当前分布式训练的总进程数（= GPU 数）
    world_size = comm.get_world_size()

    # 2. 如果 cfg 里没有显式指定 num_worker，就默认用机器全部 CPU 核心数
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()

    # 3. 把总 worker 数平均分到每张卡上（整除），后面 DataLoader 每个进程用这么多子进程
    cfg.num_worker_per_gpu = cfg.num_worker // world_size

    # 4. 保证总 batch_size 能被卡数整除，否则每张卡分到的 batch 大小不一致，训练会出错
    assert cfg.batch_size % world_size == 0

    # 5. 验证集 batch_size 同理，如果提供了的话
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0

    # 6. 测试集 batch_size 同理，如果提供了的话
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0

    # 7. 计算每张卡实际负责的 batch 大小（训练）
    cfg.batch_size_per_gpu = cfg.batch_size // world_size

    # 8. 计算每张卡实际负责的 batch 大小（验证），如果没给验证 batch_size 就默认 1
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )

    # 9. 计算每张卡实际负责的 batch 大小（测试），同理默认 1
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )

    # 10. 保证总 epoch 数是 eval_epoch 的整数倍，否则最后一个不完整 eval 周期会被跳过
    assert cfg.epoch % cfg.eval_epoch == 0

    # 11. 拿到当前进程的 rank（0 ~ world_size-1）
    rank = comm.get_rank()

    # 12. 给每个 rank 设置不同的随机种子，避免所有进程生成一模一样的数据增强；
    #    用 rank * num_worker_per_gpu 保证不同 rank、不同 worker 之间种子都不重复
    seed = None if cfg.seed is None else cfg.seed + rank * cfg.num_worker_per_gpu

    # 13. 真正设置 Python、NumPy、PyTorch 的随机种子
    set_seed(seed)

    # 14. 把更新后的 cfg 返回，供后续训练/验证代码使用
    return cfg
