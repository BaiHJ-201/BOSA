        
import torch
import torch.nn as nn
from ..utils import memory
from .base_adapter import BaseAdapter
import torch.nn.functional as F
import os
from copy import deepcopy
import numpy as np
import math
# 适用于cifar100c数据集的代码
# 其每个batch，都利用buffer数据更新统计量，每次都从源域统计量开始更新
def batch_norm(current_mean, current_var, x, weight, bias, eps):
    eps = torch.tensor([eps], dtype=current_var.dtype, device=current_var.device)
    _var = torch.sqrt(torch.maximum(current_var, eps)).view((1, -1, 1, 1)).detach()
    _mean = current_mean.view((1, -1, 1, 1))
    x_norm = (x - _mean) / _var
    if weight is not None and bias is not None:
        y = x_norm * weight.view((1, -1, 1, 1)) + bias.view((1, -1, 1, 1))
    else:
        y = x_norm
    return y
def mmd_divergence(mean1, var1, mean2, var2):
    d1 = torch.sqrt((var1 - var2) ** 2 + (mean1 - mean2) ** 2)
    return d1
def gauss_symm_kl_divergence(mean1, var1, mean2, var2, eps):
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, device=mean1.device, dtype=mean1.dtype)
    # >>> out-place ops
    dif_mean = (mean1 - mean2) ** 2
    d1 = var1 + eps + dif_mean
    d1.div_(var2 + eps)
    d2 = (var2 + eps + dif_mean)
    d2.div_(var1 + eps)
    d1.add_(d2)
    d1.div_(2.).sub_(1.)
    # d1 = (var1 + eps + dif_mean) / (var2 + eps) + (var2 + eps + dif_mean) / (var1 + eps)
    return d1

class MyBatchNorm(nn.Module):
    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.5):
        super().__init__()
        self.register_buffer("running_mean", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("running_var", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
        self.source_weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1))
        self.source_bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1))
        self.weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1))
        self.bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1))
        self.eps = 1e-5
        self.register_buffer("mu", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("sigma", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
        self.lambda_bn_d = 0.1
        self.alpha = datta_alpha

    @torch.no_grad()
    def regularize_statistics(self):
        gradient_mean = 2 * (self.mu - self.running_mean)

        target_std = torch.sqrt(self.sigma + self.eps)
        source_std = torch.sqrt(self.running_var + self.eps)
        gradient_std = 2 * target_std - 2 * source_std

        target_std = target_std - self.lambda_bn_d * gradient_std

        self.mu.copy_(self.mu - self.lambda_bn_d * gradient_mean)
        self.sigma.copy_(target_std ** 2)

    def get_soft_alignment_loss_weight(self):
        # return F.mse_loss(self.weight, self.source_weight) + F.mse_loss(self.bias, self.source_bias)
        return torch.sum((self.weight - self.source_weight) ** 2) + torch.sum((self.bias - self.source_bias) ** 2)

    def forward(self, X):
        if getattr(self, "calibrate_mode", False):

            # 当前 batch 的统计量
            buffer_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
            buffer_var = torch.mean((X - self.mu) ** 2, dim=(0, 2, 3), keepdim=True).clone()
            dist = gauss_symm_kl_divergence(
                buffer_mean, buffer_var, self.mu, self.sigma, eps=self.eps)
            adaptive_alpha = 1. - torch.exp(- 0.1 * dist.mean())
            self.alpha = adaptive_alpha.item()
            self.mu.data = self.alpha * buffer_mean + (1 - self.alpha) * self.mu.data.clone()
            self.sigma.data = self.alpha * buffer_var + (1 - self.alpha) * self.sigma.data.clone()
            # self.regularize_statistics()
            adaptive_alpha = 1. - torch.exp(- 0.1 * dist.mean())
            self.lambda_bn_d = adaptive_alpha.item()
            gradient_mean = 2 * (self.mu - self.running_mean)
            target_std = torch.sqrt(self.sigma + self.eps)
            source_std = torch.sqrt(self.running_var + self.eps)
            gradient_std = 2 * target_std - 2 * source_std
            target_std = target_std - self.lambda_bn_d * gradient_std
            self.mu.copy_(self.mu - self.lambda_bn_d * gradient_mean)
            self.sigma.copy_(target_std ** 2)

        Y = batch_norm(self.mu, self.sigma, X, self.weight, self.bias, eps=self.eps)

        return Y

class BN(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        self.alpha = cfg.ADAPTER.BN.ALPHA  
        self.theta = cfg.ADAPTER.BN.THETA 
        super(BN, self).__init__(cfg, model, optimizer)
        self.mem = memory.CSTU(capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, num_class=cfg.CORRUPTION.NUM_CLASS, lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U)
        self.has_calibrate = False
        self.lambda_bn_w = 1.0
        self.margin = 0.4*math.log(100)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.ema = None
    def forward(self, x, y):
        for _ in range(self.steps):
            outputs, ema, reset_flag = self.forward_and_adapt(x, y)
            if reset_flag:
                self.reset()
            self.ema = ema 
        return outputs

    def forward_and_adapt(self, batch_data, y):
        with torch.no_grad():
            outputs = self.model(batch_data)
            predict = torch.softmax(outputs, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)
        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
        # if self.mem.is_balanced():
        #     self.has_calibrate = True
        # if not self.has_calibrate:
        #     self.calibrate_with_buffer()
        self.calibrate_with_buffer()
        self.optimizer.zero_grad()

        outputs = self.model(batch_data)
        entropy_first = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropy_first < self.margin)[0]  
        if len(filter_ids_1) == 0:
            return outputs
        
        entropy_first = entropy_first[filter_ids_1]
        loss_first = entropy_first.mean(0)
        if self.lambda_bn_w > 0:
            l_soft_alignment = []
            for m in self.model.modules():
                if isinstance(m, MyBatchNorm):
                    l_soft_alignment.append(m.get_soft_alignment_loss_weight())
            l_soft_alignment = torch.stack(l_soft_alignment).sum()
            l_soft_alignment = l_soft_alignment * self.lambda_bn_w
        else:
            l_soft_alignment = torch.tensor(0.0).cuda()
        loss_first += l_soft_alignment
        loss_first.backward()
        self.optimizer.first_step(zero_grad=True)

        entropys2 = softmax_entropy(self.model(batch_data))
        entropys2 = entropys2[filter_ids_1]
        filter_ids_2 = torch.where(entropys2 < self.margin)
        if len(filter_ids_2) == 0:
            self.optimizer.zero_grad()
            return outputs
        
        # 计算二次损失（仅用二次筛选后的样本）
        loss_second = entropys2[filter_ids_2].mean(0)
        if self.lambda_bn_w > 0:
            l_soft_alignment = []
            for m in self.model.modules():
                if isinstance(m, MyBatchNorm):
                    l_soft_alignment.append(m.get_soft_alignment_loss_weight())
            l_soft_alignment = torch.stack(l_soft_alignment).sum()
            l_soft_alignment = l_soft_alignment * self.lambda_bn_w
        else:
            l_soft_alignment = torch.tensor(0.0).cuda()
        loss_second += l_soft_alignment
        if not np.isnan(loss_second.item()):
            ema = update_ema(self.ema, loss_second.item()) 
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)
        reset_flag = False
        if ema is not None:
            if ema < 0.1:
                print("ema < 0.1, now reset the model")
                reset_flag = True
        return outputs, ema, reset_flag
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    def replace_bn_with_custom(self, model: nn.Module, custom_bn):
        for name, module in model.named_children():
            if isinstance(module, (nn.BatchNorm2d)):
                setattr(model, name, custom_bn(module))
            else:
                self.replace_bn_with_custom(module, custom_bn)
        return model

    def configure_model(self, model: nn.Module):
        model = self.replace_bn_with_custom(model, lambda m: MyBatchNorm(m, datta_alpha = self.alpha))
        model.requires_grad_(False)
        for m in model.modules():  
            if isinstance(m, MyBatchNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
        return model
        
    def calibrate_with_buffer(self):
        imgs, ages = self.mem.get_memory()
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.calibrate_mode = True  

        if len(imgs) > 0:
            imgs = torch.stack(imgs)
            with torch.no_grad():
                _ = self.model(imgs)

        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.calibrate_mode = False

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)