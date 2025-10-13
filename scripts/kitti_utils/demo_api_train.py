import argparse
import logging

from li3d.lab.datasets import *
from li3d.lab.pipelines import *
from li3d import utils
from li3d.utils import get_module


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for training and inference')
    # parser.add_argument('--path_semantickitti',
    #                     help='path to semantiSemanticKITTI',
    #                     default='../data/semantickitti',
    #                     required=False)
    parser.add_argument('-c', '--cfg_file',
                        default='../li3d/configs/randlanet_semantickitti.yml',
                        help='path to the config file')
    # parser.add_argument('path_ckpt_randlanet',default=None,
    #                     help='path to RandLANet checkpoint')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def demo_train(cfg):
    # Initialize the training by passing parameters
    # dataset = SemanticKITTI(cfg.path_semantickitti, use_cache=True)
    #
    # model = RandLANet()
    #
    # pipeline = SemanticSegmentation(model=model, dataset=dataset, max_epoch=100)
    Pipeline = utils.get_module("pipeline", cfg.pipeline.name)
    Model = utils.get_module("model", cfg.model.name)
    Dataset = utils.get_module("dataset", cfg.dataset.name)

    dataset = Dataset(**cfg.dataset)
    model = Model(**cfg.model)
    pipeline = Pipeline(model, dataset, **cfg.pipeline)
    pipeline.run_train()


def demo_inference(args):
    Pipeline = get_module("pipeline", "SemanticSegmentation")
    Model = get_module("model", "RandLANet")
    Dataset = get_module("dataset", "SemanticKITTI")

    RandLANet = Model(ckpt_path=args.path_ckpt_randlanet)

    # Initialize by specifying config file path
    SemanticKITTI = Dataset(args.path_semantickitti, use_cache=False)

    pipeline = Pipeline(model=RandLANet, dataset=SemanticKITTI)

    # inference
    # get data
    train_split = SemanticKITTI.get_split("train")
    data = train_split.get_data(0)
    # restore weights

    # run inference
    results = pipeline.run_inference(data)
    print(results)

    # test
    pipeline.run_test()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    args = parse_args()
    if args.cfg_file is not None:
        cfg = utils.Config.load_from_file(args.cfg_file)
    else:
        raise ValueError("no cfg_file given")
    demo_train(cfg)
    # demo_inference(args)
