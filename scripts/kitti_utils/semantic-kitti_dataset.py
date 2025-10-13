# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2023/7/11 20:45
# @Author  : Leii
# @File    : semantic-kitti_dataset.py
# @Code instructions: 
# ------------------------------------------------------------------------
from os.path import join

import numpy as np
import yaml


def load_pc_kitti(pc_path):
    scan = np.fromfile(pc_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # points = scan[:, 0:3]  # get xyz
    points = scan  # get xyzi
    return points


def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def get_custom_data(pc_name, path, set_file_path):
    pc_path = join(path, 'velodyne', pc_name + '.bin')
    label_path = join(path, 'labels', pc_name + '.label')
    data_config = set_file_path
    DATA = yaml.safe_load(open(data_config, 'r'))
    remap_dict = DATA["learning_map_inv"]

    # make lookup table for mapping
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

    remap_dict_val = DATA["learning_map"]
    max_key = max(remap_dict_val.keys())
    remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut_val[list(remap_dict_val.keys())] = list(
        remap_dict_val.values())
    # load data and label
    points = load_pc_kitti(pc_path)
    labels = load_label_kitti(
        label_path, remap_lut_val).astype(np.int32)
    data = {
        'point': points[:, 0:3],
        'feat': points[:, 3:],
        'label': labels,
    }

    return data


if __name__ == "__main__":
    set_file_path = 'your/set/file/path'  # sematic-kitti.yaml
    data_path = 'your/data/path'
    pc_names = ' only the data file name'
    pcs = get_custom_data(pc_names, data_path, set_file_path)
    print(pcs['point'].shape)
    print(pcs['label'].shape)
