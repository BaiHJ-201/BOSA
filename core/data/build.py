from .datasets.common_corruption import CorruptionCIFAR
from .datasets.common_corruption import CorruptionImageNet
from .datasets.common_corruption import CorruptionMNIST
from .datasets.common_corruption import GradualCorruptionCIFAR
from .datasets.toy_dataset import ToyDataset
from .ttasampler import build_sampler
from torch.utils.data import DataLoader
from ..utils.result_precess import AvgResultProcessor
import torch
from collections import defaultdict
import random

def build_memory_buffer(cfg, dataset):
    dataset_name = cfg.CORRUPTION.DATASET.lower()
    class_num = 10 if dataset_name == "cifar10" else 100
    per_class_limit = 6 if dataset_name == "cifar10" else 1

    memory_buffer = defaultdict(lambda: {"images": [], "labels": []})

    # 尝试从 dataset 中获取 domain / label 信息
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)  # 打乱确保采样多样性

    for idx in all_indices:
        img, label, domain = dataset[idx]["image"], dataset[idx]["label"], dataset[idx]["domain"]
        domain_id = int(domain)
        label_id = int(label)

        domain_memory = memory_buffer[domain_id]
        # 计算每类已有样本数
        label_count = sum(1 for l in domain_memory["labels"] if l == label_id)
        if label_count < per_class_limit:
            domain_memory["images"].append(img)
            domain_memory["labels"].append(label_id)

        # 如果当前域已经装满，跳过
        total_needed = class_num * per_class_limit
        if len(domain_memory["labels"]) >= total_needed:
            continue

    # 转换成 tensor
    for domain_id, mem in memory_buffer.items():
        mem["images"] = torch.stack(mem["images"])
        mem["labels"] = torch.tensor(mem["labels"])

    return memory_buffer

def build_loader(cfg, ds_name, all_corruptions, all_severity):
    if ds_name == "cifar10" or ds_name == "cifar100":
        dataset_class = CorruptionCIFAR
    elif ds_name == "imagenet":
        dataset_class = CorruptionImageNet
    elif ds_name == "mnist":
        dataset_class = CorruptionMNIST
    elif ds_name == "gradualCifar10" or ds_name == "gradualCifar100":
        dataset_class = GradualCorruptionCIFAR
    elif ds_name == "toy":
        dataset_class = ToyDataset
    else:
        raise NotImplementedError(f"Not Implement for dataset: {cfg.CORRUPTION.DATASET}")

    ds = dataset_class(cfg, all_corruptions, all_severity)
    sampler = build_sampler(cfg, ds.data_source)
    loader = DataLoader(ds, cfg.TEST.BATCH_SIZE, sampler=sampler, num_workers=cfg.LOADER.NUM_WORKS)

    # === 2. 构建 memory buffer ===
    print("🔧 Building memory buffer...")
    memory_buffer = build_memory_buffer(cfg, ds)
    print("✅ Memory buffer built successfully")

    result_processor = AvgResultProcessor(ds.domain_id_to_name)
    # return loader, result_processor
    return loader, result_processor, memory_buffer
