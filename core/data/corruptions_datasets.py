import sys
import os
import json
import torch
import logging
from glob import glob
from typing import Optional, Sequence
# 添加 robustbench 根目录到 sys.path
sys.path.append(os.path.abspath("/root/TRIBE/robustbench"))

from robustbench.loaders import CustomImageFolder

import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

def create_imagenet_dataset(
    n_examples: Optional[int] = -1,
    data_dir: str = './data',
    transform=None,
    ):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruption_dir_path = os.path.join(data_dir.replace(data_dir.split('/')[-1], 'imagenet2012'), 'val')
    transform = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()])
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    # load imagenet class to id mapping from robustbench
    with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
        class_to_idx = json.load(f)


    file_path = os.path.join("datasets", "imagenet_list", "imagenet_val_ids_50k.txt")


    # load file containing file ids
    with open(file_path, 'r') as f:
        fnames = f.readlines()
    # print(len(fnames))
    item_list = []
    item_list += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
    dataset_test.samples = item_list

    return dataset_test