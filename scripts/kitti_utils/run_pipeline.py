import argparse
import logging
import os
import json_files
import pprint
import sys
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch import multiprocessing

from li3d.lab.datasets import *
from li3d import utils
from li3d.lab.pipelines import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('-c', '--cfg_file', help='path to the config file',
                        default='../li3d/configs/pvann_semantickitti.yml', )
    parser.add_argument('-m', '--model', help='network model')
    parser.add_argument('-p', '--pipeline', help='pipeline',
                        default='SemanticSegmentation')
    parser.add_argument('-d', '--dataset', help='dataset')
    parser.add_argument('--cfg_model', help='path to the model\'s config file')
    parser.add_argument('--cfg_pipeline', help='path to the pipeline\'s config file')
    parser.add_argument('--cfg_dataset', help='path to the dataset\'s config file')
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--ckpt_path', help='path to the checkpoint')
    parser.add_argument('--device', help='devices to run the pipeline',
                        default='cuda')
    parser.add_argument('--device_ids', nargs='+', help='cuda device list',
                        default=['0'])
    parser.add_argument('--split', help='train or test',
                        default='train')
    parser.add_argument('--mode', help='additional mode',
                        default=None)
    parser.add_argument('--max_epochs', help='number of epochs',
                        default=None)
    parser.add_argument('--batch_size', help='batch size',
                        default=None)
    parser.add_argument('--main_log_dir', help='the dir to save logs and models')
    parser.add_argument('--seed', help='random seed',
                        default=0, type=int)
    parser.add_argument('--nodes', help='number of nodes',
                        default=1, type=int)
    parser.add_argument('--node_rank',
                        help='ranking within the nodes, default: 0. To get from'
                             ' the environment, enter the name of an env var eg: '
                             '"SLURM_NODEID".',
                        default="0",
                        type=str)
    parser.add_argument('--host', help='Host for distributed training, default: localhost',
                        default='localhost')
    parser.add_argument('--port', help='port for distributed training, default: 12355',
                        default='12355')
    parser.add_argument('--backend', help='backend for distributed training. One of (nccl, gloo)}, default: gloo',
                        default='gloo')

    args, unknown = parser.parse_known_args()  # unknown表示没有被解析的参数
    try:
        args.node_rank = int(args.node_rank)
    except ValueError:  # str => get from environment
        args.node_rank = int(os.environ[args.node_rank])

    parser_extra = argparse.ArgumentParser(description='Extra arguments')
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser_extra.add_argument(arg)
    args_extra = parser_extra.parse_args(unknown)

    print("regular arguments")
    print(yaml.dump(vars(args)))

    print("extra arguments")
    print(yaml.dump(vars(args_extra)))

    return args, vars(args_extra)


def main():
    cmd_line = ' '.join(sys.argv[:])
    args, extra_dict = parse_args()

    args.device, args.device_ids = utils.convert_device_name(
        args.device, args.device_ids)
    rng = np.random.default_rng(args.seed)

    if args.cfg_file is not None:
        cfg = utils.Config.load_from_file(args.cfg_file)

        Pipeline = utils.get_module("pipeline", cfg.pipeline.name)
        Model = utils.get_module("model", cfg.model.name)
        Dataset = utils.get_module("dataset", cfg.dataset.name)

        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
            utils.Config.merge_cfg_file(cfg, args, extra_dict)

        if args.mode is not None:
            cfg_dict_model["mode"] = args.mode
        if args.max_epochs is not None:
            cfg_dict_pipeline["max_epochs"] = args.max_epochs
        if args.batch_size is not None:
            cfg_dict_pipeline["batch_size"] = args.batch_size

        cfg_dict_dataset['seed'] = rng
        cfg_dict_model['seed'] = rng
        cfg_dict_pipeline['seed'] = rng

        cfg_dict_pipeline["device"] = args.device
        cfg_dict_pipeline["device_ids"] = args.device_ids

    else:
        if (args.pipeline and args.model and args.dataset) is None:
            raise ValueError("Please specify pipeline, model, and dataset " +
                             "if no cfg_file given")

        Pipeline = utils.get_module("pipeline", args.pipeline)
        Model = utils.get_module("model", args.model)
        Dataset = utils.get_module("dataset", args.dataset)

        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
            utils.Config.merge_module_cfg_file(args, extra_dict)

        cfg_dict_dataset['seed'] = rng
        cfg_dict_model['seed'] = rng
        cfg_dict_pipeline['seed'] = rng

    with open(Path(__file__).parent / 'README.md', 'r') as freadme:
        readme = freadme.read()

    cfg_tb = {
        'readme': readme,
        'cmd_line': cmd_line,
        'dataset': pprint.pformat(cfg_dict_dataset, indent=2),
        'model': pprint.pformat(cfg_dict_model, indent=2),
        'pipeline': pprint.pformat(cfg_dict_pipeline, indent=2)
    }
    # # 使用 with 语句打开文件
    # with open("./cfg_tb.json_files", "w") as f:
    #     json_files.dump(cfg_tb, f, ensure_ascii=False, indent=2)
    args.cfg_tb = cfg_tb
    args.distributed = args.device != 'cpu' and len(
        args.device_ids) > 1

    if not args.distributed:
        dataset = Dataset(**cfg_dict_dataset)  # 这里dataset被实例化
        # model = Model(**cfg_dict_model, mode=args.mode)
        pipeline = Pipeline(model=Model, model_cfg=cfg_dict_model,
                            model_mode=args.mode, dataset=dataset,
                            **cfg_dict_pipeline)

        pipeline.cfg_tb = cfg_tb

        if args.split == 'test':
            pipeline.run_test()
        else:
            pipeline.run_train()

    else:
        mp.spawn(main_worker,
                 args=(Dataset, Model, Pipeline, cfg_dict_dataset,
                       cfg_dict_model, cfg_dict_pipeline, args),
                 nprocs=len(args.device_ids))


def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.host
    os.environ['MASTER_PORT'] = args.port

    # initialize the process group
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main_worker(local_rank, Dataset, Model, Pipeline, cfg_dict_dataset,
                cfg_dict_model, cfg_dict_pipeline, args):
    rank = args.node_rank * len(args.device_ids) + local_rank
    world_size = args.nodes * len(args.device_ids)
    setup(rank, world_size, args)

    cfg_dict_dataset['rank'] = rank
    cfg_dict_model['rank'] = rank
    cfg_dict_pipeline['rank'] = rank

    rng = np.random.default_rng(args.seed + rank)
    cfg_dict_dataset['seed'] = rng
    cfg_dict_model['seed'] = rng
    cfg_dict_pipeline['seed'] = rng

    device = f"cuda:{args.device_ids[local_rank]}"
    print(
        f"local_rank = {local_rank}, rank = {rank}, world_size = {world_size},"
        f" gpu = {device}")

    cfg_dict_model['device'] = device
    cfg_dict_pipeline['device'] = device

    dataset = Dataset(**cfg_dict_dataset)
    model = Model(**cfg_dict_model, mode=args.mode)
    pipeline = Pipeline(model,
                        dataset,
                        distributed=args.distributed,
                        **cfg_dict_pipeline)

    pipeline.cfg_tb = args.cfg_tb

    if args.split == 'test':
        if rank == 0:
            pipeline.run_test()
    else:
        pipeline.run_train()

    cleanup()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    try:
        multiprocessing.set_start_method('spawn')
    except:
        log.info('can not use multiprocess...')
    sys.exit(main())
