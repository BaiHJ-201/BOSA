import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
# from batch_norm import BatchNorm
import json
import os
import tqdm
import PIL
import torchvision.transforms as transforms
import numpy as np

import torch.nn.functional as F
from .base_adapter import BaseAdapter
from ..data.data_loading import get_source_loader

logger = logging.getLogger(__name__)

def to_serializable(obj):
    if torch.is_tensor(obj):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_serializable(v) for v in obj)
    else:
        return obj


def print_dict_shapes(d, indent=0, max_list_items=3):
    prefix = " " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{prefix}{k}:")
            print_dict_shapes(v, indent + 2, max_list_items=max_list_items)

    elif isinstance(d, torch.Tensor):
        print(f"{prefix}Tensor, shape={tuple(d.shape)}, dtype={d.dtype}")

    elif isinstance(d, np.ndarray):
        print(f"{prefix}ndarray, shape={d.shape}, dtype={d.dtype}")

    elif isinstance(d, list):
        print(f"{prefix}list of length {len(d)}")
        for i, item in enumerate(d[:max_list_items]):  # 只打印前几项，防止太长
            print(f"{prefix}  [{i}]:")
            print_dict_shapes(item, indent + 4, max_list_items=max_list_items)
        if len(d) > max_list_items:
            print(f"{prefix}  ... (only first {max_list_items} shown)")

    else:
        print(f"{prefix}{type(d)}")

def check_source_loader(source_dataset, source_loader, name="source"):
    print("=" * 60)
    print(f"🔍 Checking {name} dataset & dataloader")
    print(f"- Number of images in dataset: {len(source_dataset)}")
    print(f"- Number of batches (with batch_size={source_loader.batch_size}): {len(source_loader)}")
    print(f"- First batch shape: ")
    for images, labels in source_loader:
        print(f"  images: {images.shape}, labels: {labels.shape}")
        break  # 只看第一个 batch
    print("=" * 60)

def batch_norm(mean, var, X, weight, bias, eps):

    X_hat = (X - mean) / torch.sqrt(var + eps)

    Y = weight * X_hat + bias  # Scale and shift

    means = torch.mean(Y.clone(), dim=(2, 3))
    vars = torch.mean((Y.clone() - means.unsqueeze(2).unsqueeze(3)) ** 2, dim=(2, 3))

    return Y, (means, vars)


class MyBatchNorm(nn.Module):

    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.0):
        super().__init__()

        try:
            num_features = bn_init.num_features
        except AttributeError:
            num_features = bn_init.num_feature

        self.register_buffer("running_mean", bn_init.running_mean.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("running_var", bn_init.running_var.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

        self.weight = nn.Parameter(bn_init.weight.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.bias = nn.Parameter(bn_init.bias.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

        self.eps = 1e-5
        self.register_buffer("weight_init", bn_init.weight.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("bias_init", bn_init.bias.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("mu", torch.ones(1, num_features, 1, 1))
        self.register_buffer("sigma", torch.zeros(1, num_features, 1, 1))

        self.stat = None
        self.reset_mean = True
        self.alpha = datta_alpha

    def reset_statistic(self):
        self.reset_mean = True
        # self.weight.data = self.weight_init.clone().detach()
        # self.bias.data = self.bias_init.clone().detach()

    def set_alpha(self, alpha):
        self.reset_mean = True
        self.alpha = alpha

    def forward(self, X):
        if self.reset_mean:
            self.reset_mean = False
            alpha = self.alpha
            self.mu.data = (1 - alpha) * torch.mean(X, dim=(0, 2, 3), keepdim=True).clone() + alpha * self.running_mean.data.clone()
            self.sigma.data = (1 - alpha) * torch.mean((X - self.mu) ** 2, dim=(0, 2, 3), keepdim=True).clone() + alpha * self.running_var.data.clone()

        Y, self.stat = batch_norm(self.mu, self.sigma, X, self.weight, self.bias, eps=self.eps)

        return Y


class DATTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(DATTA, self).__init__(cfg, model, optimizer)
        self.alpha = cfg.ADAPTER.DATTA.ALPHA  # Equation 6,7
        self.theta = cfg.ADAPTER.DATTA.THETA  # Equation 8
        self.model = model

        self.stat_outputs = {}

        dir_source_distri = "./ckpt/source_distribution/"
        self.path_source_distri = dir_source_distri + cfg.CORRUPTION.DATASET.split('_')[0] + f"_{cfg.MODEL.ARCH}.json"
        if os.path.exists(self.path_source_distri):
            with open(self.path_source_distri) as file:
                self.source_distri = json.load(file)
        else:
            self.source_dataset, self.source_dataloader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                                   root_dir=cfg.DATA_DIR,
                                                   batch_size=128, ckpt_path=cfg.CKPT_PATH,
                                                   workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
            check_source_loader(self.source_dataset, self.source_dataloader, name="source")
            if not os.path.exists(dir_source_distri):
                os.makedirs(dir_source_distri, exist_ok=True)
            logger.info("Pre-compute source distribution statistics...")
            self.precompute_source_distri()
            with open(self.path_source_distri) as file:
                self.source_distri = json.load(file)

    @torch.no_grad()
    def precompute_source_distri(self):

        def avg_batch(data, layer=0):
            result = {'means': [], 'vars': []}
            for name in ['means', 'vars']:
                result[name] = [sum(values) / len(values) for values in zip(*data[name][layer])]
                result[name] = torch.tensor(result['means'], dtype=torch.float)
            return result

        model = self.model
        avg_per_batch = {'means': [], 'vars': []}
        batch_num = 0
        for data in tqdm.tqdm(self.source_dataloader):
            batch_num += 1
            x = data[0].cuda()
            for m in model.modules():
                if isinstance(m, MyBatchNorm):
                    m.reset_statistic()
            _ = model(x)

            stat = list(self.stat_outputs.values())
            
            stats = [item for pair in stat for item in pair]

            # [("m1", "v1"), ("m2", "v2"),("m3", "v3"),] ->["m1", "v1", "m2", "v2", "m3", "v3"]
            # m1, v1, m2, v2分别是第一、第二个改良BN层的统计量，m\v都是[Batch,Channl]的张量
            # print(len(stats))  # 53 * 2 len if imagenet(resnet50), 31 * 2 if cifar100C, 25 * 2 if cifar10c, 170*2 if res2net
            for i in range(len(stats)):
                stats[i] = stats[i].cpu().tolist()
            stats_one_batch = {'means': [], 'vars': []}
            for i, stat in enumerate(stats):
                if i % 2 == 0:
                    # 每个层的一个batch内的特征的均值
                    stats_one_batch['means'].append(stat)
                else:
                    # 每个层的一个batch内的特征的方差
                    stats_one_batch['vars'].append(stat)
            for i in range(len(stats_one_batch['means'])):
                # 对每个层的batch求均值
                result = avg_batch(stats_one_batch, i)
                # if i == 0 or i == 1:
                #     print_dict_shapes(result)
                # 对每个batch的数据就相加
                if len(avg_per_batch['means']) == len(stats) / 2:
                    avg_per_batch['means'][i] = avg_per_batch['means'][i] + result['means']
                    avg_per_batch['vars'][i] = avg_per_batch['vars'][i] + result['vars']
                else:  # for first batch
                    avg_per_batch['means'].append(result['means'])
                    avg_per_batch['vars'].append(result['vars'])
        print_dict_shapes(avg_per_batch)
        print(batch_num)
        avg_all_batch = {'means': [], 'vars': []}
        for layer in range(len(avg_per_batch['means'])):
            avg_all_batch['means'].append(avg_per_batch['means'][layer] / batch_num)
            avg_all_batch['vars'].append(avg_per_batch['vars'][layer] / batch_num)
        assert len(avg_all_batch['vars']) == len(stats) / 2
        assert len(avg_all_batch['means']) == len(stats) / 2
        
        with open(self.path_source_distri, 'w') as f:
            f.write(json.dumps(
                {
                    "stats": to_serializable(avg_all_batch)
                },
                indent=4,
            ))
    def replace_bn_with_custom(self, model: nn.Module, custom_bn):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, custom_bn(module))
            else:
                self.replace_bn_with_custom(module, custom_bn)
        return model
    
    # define hook
    def get_stat_output(self, name):
        def hook(module, input, output):
            self.stat_outputs[name] = module.stat  # Collect the statistic output
        return hook
    
    # attach hook to get the statistic during forward pass
    def attach_hooks_to_custom_bn(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, MyBatchNorm):
                module.register_forward_hook(self.get_stat_output(name))

    def configure_model(self, model: nn.Module):
        model = self.replace_bn_with_custom(model, MyBatchNorm)
        self.attach_hooks_to_custom_bn(model)
        model.train()
        model.requires_grad_(False)
        for m in model.modules():  # 25, 31 ,53
            if isinstance(m, MyBatchNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True

        return model

    def reset(self):
        # print(type(self.model_states), self.optimizer_state)
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

        # reset the initial statistics of BNs
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """
        Forward and adapt model on batch of data.
        """

        outputs = model(x)
        stat = list(self.stat_outputs.values())

        # confidence threshold
        confidence = torch.softmax(outputs, dim=1)
        outputs_above_threshold = []
        for j in range(confidence.shape[0]):
            if torch.max(confidence[j]) > self.theta:
                outputs_above_threshold.append(outputs[j])

        loss_stat = stat_loss(stat, self.source_distri['stats'])
        if len(outputs_above_threshold) != 0:
            outputs_above_threshold = torch.stack(outputs_above_threshold)
            loss_em = softmax_entropy(outputs_above_threshold).mean(0)
            loss = loss_stat + loss_em
        else:
            loss = loss_stat

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.stat_outputs = {}
        return outputs

    def forward(self, x):
        outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs


    def collect_params(self, model: nn.Module):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, MyBatchNorm):
                for np, p in m.named_parameters():
                    if p.requires_grad == True:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        print("collect:", names)
        return params, names

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


l1loss = nn.L1Loss(reduction='mean')
def stat_loss(stat, source_stat):
    for i in range(len(stat)):
        bs = stat[0][0].shape[0]
        if i == 0:
            losses_aff_mean = l1loss(stat[i][0], torch.tensor(source_stat['means'][i]).unsqueeze(0).repeat(bs, 1).cuda())
            losses_aff_var = l1loss(stat[i][1], torch.tensor(source_stat['vars'][i]).unsqueeze(0).repeat(bs, 1).cuda())
        else:
            losses_aff_mean = torch.add(losses_aff_mean, l1loss(stat[i][0], torch.tensor(source_stat['means'][i]).unsqueeze(0).repeat(bs, 1).cuda()))
            losses_aff_var = torch.add(losses_aff_var, l1loss(stat[i][1], torch.tensor(source_stat['vars'][i]).unsqueeze(0).repeat(bs, 1).cuda()))

    return losses_aff_mean + losses_aff_var








