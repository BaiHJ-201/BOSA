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
    per_class_limit = 6 if dataset_name == "cifar10" else 2

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
# import torch
# import random
# import numpy as np
# from collections import defaultdict

# def build_memory_buffer(cfg, dataset):
#     dataset_name = cfg.CORRUPTION.DATASET.lower()
#     dataset_name = dataset_name.lower()

#     if dataset_name == "cifar10":
#         class_num = 10
#         per_class_limit = 6  # 保持原逻辑
#     else:
#         class_num = 100
#         total_images = 200  # 目标总数 200

#         # 计算比例 N_i
#         gamma = 0.1
#         img_max = 100
#         Ni_raw = np.array([img_max * (gamma ** (i / (class_num - 1))) for i in range(class_num)])
#         # 归一化缩放
#         Ni_scaled = Ni_raw / Ni_raw.sum() * total_images
#         Ni_per_class = np.round(Ni_scaled).astype(int)

#         # 确保总数精确为 200（修正舍入误差）
#         diff = total_images - Ni_per_class.sum()
#         Ni_per_class[0:abs(diff)] += np.sign(diff)

#     memory_buffer = defaultdict(lambda: {"images": [], "labels": []})

#     # 打乱确保采样多样性
#     all_indices = list(range(len(dataset)))
#     random.shuffle(all_indices)

#     # 每个域单独构建
#     for idx in all_indices:
#         img, label, domain = dataset[idx]["image"], dataset[idx]["label"], dataset[idx]["domain"]
#         domain_id = int(domain)
#         label_id = int(label)

#         domain_memory = memory_buffer[domain_id]

#         if dataset_name == "cifar10":
#             label_count = sum(1 for l in domain_memory["labels"] if l == label_id)
#             if label_count < per_class_limit:
#                 domain_memory["images"].append(img)
#                 domain_memory["labels"].append(label_id)
#             if len(domain_memory["labels"]) >= class_num * per_class_limit:
#                 continue
#         else:
#             # cifar100 模式：按 Ni_per_class 控制每类数量
#             label_limit = Ni_per_class[label_id]
#             label_count = sum(1 for l in domain_memory["labels"] if l == label_id)
#             if label_count < label_limit:
#                 domain_memory["images"].append(img)
#                 domain_memory["labels"].append(label_id)
#             if len(domain_memory["labels"]) >= total_images:
#                 continue

#     # 转换成 tensor
#     for domain_id, mem in memory_buffer.items():
#         mem["images"] = torch.stack(mem["images"])
#         mem["labels"] = torch.tensor(mem["labels"])

#     return memory_buffer
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

    result_processor = AvgResultProcessor(ds.domain_id_to_name)
    return loader, result_processor
    # return loader, result_processor, memory_buffer