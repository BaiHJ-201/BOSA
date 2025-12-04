# 伪代码：生成 cov_{model}.npy 的核心流程
import torch
import numpy as np
import logging
import torch
import argparse
import sys
import os
import inspect
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
sys.path.insert(0, os.path.join(current_dir, 'tinytl', 'once-for-all'))
from core.configs import cfg
from core.utils import *
from core.model import build_model
from core.data import build_loader
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle
from sklearn.metrics import confusion_matrix
import numpy as np

import time

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from tinytl.memory_cost_profiler import profile_memory_cost
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar100_loader(n_examples=500, batch_size=64, shuffle=True, num_workers=4, data_dir='./datasets', model=None):
    """
    加载CIFAR-100数据集的指定数量随机样本，并返回DataLoader
    根据模型自动选择对应的标准化参数
    
    参数:
        n_examples: 要加载的样本数量
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的进程数
        data_dir: 数据存储路径
        model: 模型实例，用于获取标准化参数
    
    返回:
        DataLoader: 包含指定数量随机样本的DataLoader
    """
    # 根据模型类型选择对应的标准化参数
    if hasattr(model, 'mu') and hasattr(model, 'sigma'):
        # 从模型中获取预定义的均值和标准差（已转换为列表格式）
        CIFAR100_MEAN = model.mu.squeeze().tolist()
        CIFAR100_STD = model.sigma.squeeze().tolist()
    else:
        # 默认使用CIFAR-100标准参数（若模型无预定义）
        CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
        CIFAR100_STD = [0.2675, 0.2565, 0.2761]
    
    # 数据预处理（仅ToTensor，标准化在模型内部完成的情况不需要额外处理）
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 注意：如果模型forward中已经做了标准化（如Hendrycks2020AugMixResNeXtNet），这里不需要再Normalize
        # 仅当模型未内置标准化时才添加下面这行
        # transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    ])
    
    # 加载完整训练集
    full_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 随机选择n_examples个样本的索引
    total_samples = len(full_dataset)
    if n_examples > total_samples:
        raise ValueError(f"请求的样本数({n_examples})超过数据集总样本数({total_samples})")
    
    random_indices = np.random.choice(total_samples, size=n_examples, replace=False)
    subset_dataset = Subset(full_dataset, random_indices)
    
    # 创建DataLoader
    data_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return data_loader
def extract_features(model, x):
    """
    针对不同模型提取特征（修复torch.relu的inplace参数错误）
    """
    if hasattr(model, 'forward_features'):
        # 如果模型有forward_features方法，直接调用
        return model.forward_features(x)
    else:
        # 针对Hendrycks2020AugMixResNeXtNet的特征提取逻辑
        # 1. 模型内部的标准化
        x = (x - model.mu) / model.sigma
        # 2. 特征提取部分（对应ResNeXt的backbone）
        x = model.conv_1_3x3(x)
        # 修正：torch.relu没有inplace参数，移除该参数或改用nn.ReLU()
        x = torch.relu(model.bn_1(x))  # 去掉inplace=True
        x = model.stage_1(x)
        x = model.stage_2(x)
        x = model.stage_3(x)
        x = model.avgpool(x)  # 全局平均池化后的特征
        return x.view(x.size(0), -1)  # 展平为特征向量
def testTimeAdaptation(cfg):
    model = build_model(cfg).eval()
    # 1. 加载预训练模型和少量源数据（如 ImageNet 随机采样 500 张图）
    source_dataloader = get_cifar100_loader(n_examples=500, model=model)  # 仅采样少量源数据

    # 2. 提取特征并统计方差
    features = []
    with torch.no_grad():
        for x, _ in source_dataloader:
            x = x.to(next(model.parameters()).device)  # 确保输入与模型在同一设备
            # 提取backbone特征（使用适配后的提取函数）
            z = extract_features(model, x)
            features.append(z.cpu().numpy())

    # 3. 计算每个特征维度的方差（按特征维度统计，得到 1D 数组）
    features = np.concatenate(features, axis=0)  # shape: [500, feature_dim]
    feature_var = np.var(features, axis=0)       # shape: [feature_dim]（每个维度的方差）

    # 4. 保存为 npy 文件（命名格式：cov_{model}.npy，如 cov_resnet50.npy）
    np.save(f'utils/Hendrycks2020AugMix_WRN.npy', feature_var)

def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        '-ocfg',
        '--order-config-file',
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str)
    parser.add_argument(
        '-pcfg',
        '--protocol-config-file',
        metavar="FILE",
        default="",
        help="path to protocol config file",
        type=str)
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if not args.order_config_file == "":
        cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_file(args.protocol_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ds = cfg.CORRUPTION.DATASET
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA:{ds:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TTA', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    logger.info(f"Loaded configuration file: \n"
                f"\tadapter: {args.adapter_config_file}\n"
                f"\tdataset: {args.dataset_config_file}\n"
                f"\torder: {args.order_config_file}")
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)

    testTimeAdaptation(cfg)


if __name__ == "__main__":
    main()
