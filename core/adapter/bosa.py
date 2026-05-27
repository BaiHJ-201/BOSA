import copy
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states

from ..utils import memory
from .base_adapter import BaseAdapter
from ..utils.custom_transforms import get_tta_transforms

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

def batch_norm(mean, var, X, weight, bias, eps):
    X_hat = (X - mean) / torch.sqrt(var + eps)
    Y = weight * X_hat + bias  # Scale and shift
    return Y

class BOSA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):  
        self.ema_decay = cfg.ADAPTER.BOSA.EMA_DECAY
        self.update_frequency = max(1, int(cfg.ADAPTER.BOSA.UPDATE_FREQUENCY))
        self.current_instance = 0
        MyBatchNorm.reset_runtime_state()
        super(BOSA, self).__init__(cfg, model, optimizer)
        self.mem = memory.CSTU(capacity=self.cfg.ADAPTER.BOSA.MEMORY_SIZE, num_class=cfg.CORRUPTION.NUM_CLASS, lambda_t=cfg.ADAPTER.BOSA.LAMBDA_T, lambda_u=cfg.ADAPTER.BOSA.LAMBDA_U)
        
        self.teacher = copy.deepcopy(self.model)
        self.transform = get_tta_transforms(cfg)
        
        # Flag teacher BN modules to prevent them from modifying global metrics
        for m in self.teacher.modules():
            if isinstance(m, MyBatchNorm):
                m.is_teacher = True
                
        for p in self.teacher.parameters():
            p.requires_grad = False
            p.detach_()
        return

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        with torch.no_grad():
            self.teacher.eval()
            teacher_outputs = self.teacher(batch_data)
            entropy = self.self_softmax_entropy(teacher_outputs)
            teacher_probs = torch.softmax(teacher_outputs, dim=1)
            pseudo_label = torch.argmax(teacher_probs, dim=1)
        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)
        return teacher_outputs

    def update_model(self, model, optimizer):
        if optimizer is None or self.mem.get_occupancy() == 0:
            return

        model.train()
        self.teacher.train()
        imgs, _ = self.mem.get_memory()
        imgs = torch.stack(imgs)
        strong_aug = self.transform(imgs)
        
        # Roll over the max score from the previous batch to normalize the current batch's on-the-fly adaptive sparsity
        MyBatchNorm.previous_batch_max_score = max(1e-8, MyBatchNorm.current_batch_max_score)
        MyBatchNorm.current_batch_max_score = 1e-8

        ema_out = self.teacher(imgs)
        stu_out = model(strong_aug)
        loss = (softmax_entropy(stu_out, ema_out)).mean()
        
        optimizer.zero_grad()
        if has_accum_bn_grad(model):
            loss.backward()
            optimizer.step()
        self.update_teacher()

    @torch.no_grad()
    def update_teacher(self):
        for t_params, s_params in zip(self.teacher.parameters(), self.model.parameters()):
            t_params.data.mul_(self.ema_decay).add_(s_params.data * (1 - self.ema_decay))
            
    def reset(self):
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()
       
    @staticmethod
    def self_softmax_entropy(x):
        return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1)

    def replace_bn_with_custom(self, model: nn.Module, custom_bn):
        for name, module in model.named_children():
            if isinstance(module, (nn.BatchNorm2d)):
                setattr(model, name, custom_bn(module))
            else:
                self.replace_bn_with_custom(module, custom_bn)
        return model

    def configure_model(self, model: nn.Module):
        model = self.replace_bn_with_custom(model, lambda m: MyBatchNorm(m))
        model.requires_grad_(False)
        for m in model.modules():  
            if isinstance(m, MyBatchNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
        return model

@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def has_accum_bn_grad(model):
    """Return True, if at least one param has grad."""
    all_matched = True
    has_acc_bn = False
    for m in model.modules():
        if isinstance(m, MyBatchNorm):
            has_acc_bn = True
            if not m.full_matched:
                all_matched = False
                break
    if has_acc_bn and all_matched:
        return False
    return True
