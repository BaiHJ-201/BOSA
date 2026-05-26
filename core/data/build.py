from .datasets.common_corruption import CorruptionCIFAR
from .datasets.common_corruption import CorruptionImageNet
from .datasets.common_corruption import ImageNetV2
from .ttasampler import build_sampler
from torch.utils.data import DataLoader


def build_loader(cfg, ds_name, all_corruptions, all_severity):
    if ds_name in ["cifar10", "cifar100"]:
        dataset_class = CorruptionCIFAR
    elif ds_name == "imagenetv2":
        dataset_class = ImageNetV2
    elif ds_name == "imagenet":
        dataset_class = CorruptionImageNet
    else:
        raise NotImplementedError(f"Not Implement for dataset: {cfg.CORRUPTION.DATASET}")

    if ds_name == "imagenetv2":
        ds = dataset_class(cfg)
    else:
        ds = dataset_class(cfg, all_corruptions, all_severity)
    
    sampler = build_sampler(cfg, ds.data_source)
    
    return DataLoader(ds, cfg.TEST.BATCH_SIZE, sampler=sampler, num_workers=cfg.LOADER.NUM_WORKS)
