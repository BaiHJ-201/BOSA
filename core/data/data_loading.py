import os
import logging
import random
import numpy as np
import time
import json

import torch
import torchvision
import torchvision.transforms as transforms
from ..configs.defaults import complete_data_dir_path

from .imagenet_subsets import create_imagenet_subset
from .corruptions_datasets import create_imagenet_dataset

logger = logging.getLogger(__name__)


def get_transform(dataset_name):
    """
    Get transformation pipeline
    Note that the data normalization is done inside of the model
    :param dataset_name: Name of the dataset
    :param adaptation: Name of the adaptation method
    :return: transforms
    """

    # create non-method specific transformation
    if dataset_name in {"cifar10", "cifar100"}:
        transform = transforms.Compose([transforms.ToTensor()])
    elif dataset_name in {"cifar10_c", "cifar100_c"}:
        transform = None
    elif dataset_name == "imagenet_c":
        # note that ImageNet-C is already resized and centre cropped
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        # use classical ImageNet transformation procedure
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])

    return transform

def get_source_loader(dataset_name, root_dir, batch_size, train_split=False, ckpt_path=None, num_samples=None, percentage=1.0, workers=4):
    # create the name of the corresponding source dataset
    dataset_name = dataset_name.split("_")[0] if dataset_name in {"cifar10_c", "cifar100_c", "imagenet_c", "imagenet_k"} else dataset_name

    # complete the root path to the full dataset path
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)

    # setup the transformation pipeline
    transform = get_transform(dataset_name)

    # create the source dataset
    if dataset_name == "cifar10":
        source_dataset = torchvision.datasets.CIFAR10(root=root_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name == "cifar100":
        source_dataset = torchvision.datasets.CIFAR100(root=root_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
    elif dataset_name == "imagenet":
        try:
            split = "train" if train_split else "val"
            source_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                           split=split,
                                                           transform=transform)
        except RuntimeError:
            source_dataset = create_imagenet_dataset(n_examples=-1,
                                                   data_dir=data_dir,
                                                   transform=transform)
    elif dataset_name in {"imagenet_r", "imagenet_a", "imagenet_d"}:
        split = "train" if train_split else "val"
        data_dir = complete_data_dir_path(root=root_dir, dataset_name="imagenet")
        source_dataset = create_imagenet_subset(data_dir=data_dir,
                                                dataset_name=dataset_name,
                                                split=split,
                                                transform=transform)
    else:
        raise ValueError("Dataset not supported.")

    if percentage < 1.0 or num_samples:    # reduce the number of source samples
        if dataset_name in {"cifar10", "cifar100"}:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)

        logger.info(f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")

    # create the source data loader
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                drop_last=False)
    logger.info(f"Number of images and batches in source loader: #img = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_dataset, source_loader