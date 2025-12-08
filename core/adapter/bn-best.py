import torch
import torch.nn as nn
from ..utils import memory
from .base_adapter import BaseAdapter
import torch.nn.functional as F
import os
import math
import copy
import torchvision.transforms as transforms
from . import my_transforms
import PIL
# 适用于cifar100c数据集的代码
# 其每个batch，都利用buffer数据更新统计量，每次都从源域统计量开始更新
def batch_norm(mean, var, X, weight, bias, eps):

    X_hat = (X - mean) / torch.sqrt(var + eps)

    Y = weight * X_hat + bias  # Scale and shift

    return Y
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
def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False, dataset='cifar'):
    img_shape = (32, 32, 3) if 'cifar' in dataset else (224, 224, 3)
    print('img_shape in cotta transform', img_shape)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms
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

        target_std = target_std - self.alpha * gradient_std

        self.mu.copy_(self.mu - self.alpha * gradient_mean)
        self.sigma.copy_(target_std ** 2)
      
    def get_soft_alignment_loss_weight(self):
        # return F.mse_loss(self.weight, self.source_weight) + F.mse_loss(self.bias, self.source_bias)
        return torch.sum((self.weight - self.source_weight) ** 2) + torch.sum((self.bias - self.source_bias) ** 2)

    def forward(self, X):
        if getattr(self, "calibrate_mode", False):

            # 当前 batch 的统计量
            buffer_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
            buffer_var = torch.var(X, dim=(0, 2, 3), keepdim=True, unbiased=True).clone()
            dist = gauss_symm_kl_divergence(
                buffer_mean, buffer_var, self.mu, self.sigma, eps=self.eps)
          
            adaptive_alpha = 1. - torch.exp(- 1.0 * dist.mean())
            self.alpha = adaptive_alpha.item()
            self.mu.data = self.alpha * buffer_mean + (1 - self.alpha) * self.mu.data.clone()
            self.sigma.data = self.alpha * buffer_var + (1 - self.alpha) * self.sigma.data.clone()
            # self.regularize_statistics()
            adaptive_alpha = 1. - torch.exp(- 0.1 * dist.mean())
            self.alpha = adaptive_alpha.item()
            gradient_mean = 2 * (self.mu - self.running_mean)

            target_std = torch.sqrt(self.sigma + self.eps)
            source_std = torch.sqrt(self.running_var + self.eps)
            gradient_std = 2 * target_std - 2 * source_std

            target_std = target_std - self.alpha * gradient_std

            self.mu.copy_(self.mu - self.alpha * gradient_mean)
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
        self.margin = 0.4
        self.lambda_bn_w = 0.0
        self.ema_decay = 0.999
        self.teacher = copy.deepcopy(self.model)
        self.model.eval()
        self.teacher.eval()
        self.transform = get_tta_transforms(dataset='cifar')
        for p in self.teacher.parameters():
            p.requires_grad = False
            p.detach_()
        return

    def forward_and_adapt(self, batch_data):
        batch_size = len(batch_data)
        # outputs = self.model(batch_data)
        with torch.no_grad():
            teacher_outputs = self.teacher(batch_data)
            # teacher_outputs = self.teacher(batch_data)
            teacher_probs = torch.softmax(teacher_outputs, dim=1)
            pseudo_label = torch.argmax(teacher_probs, dim=1)
            entropy = torch.sum(- teacher_probs * torch.log(teacher_probs + 1e-6), dim=1)
            # pseudo_acc = (pseudo_label == y).float().mean().item()
            # print(f"[Pseudo Label Accuracy] {pseudo_acc * 100:.2f}%")
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
        # imgs, ages = self.mem.get_memory()
        # memory_size = len(imgs)
        # for m in self.model.modules():
        #     if isinstance(m, MyBatchNorm):
        #         m.calibrate_mode = True  

        # if len(imgs) > 0:
        #     imgs = torch.stack(imgs)
        #     with torch.no_grad():
        #         _ = self.model(imgs)

        # for m in self.model.modules():
        #     if isinstance(m, MyBatchNorm):
        #         m.calibrate_mode = False
        strong_aug = self.transform(batch_data)
        with torch.no_grad():
            for m in self.teacher.modules():
                if isinstance(m, MyBatchNorm):
                    m.calibrate_mode = True  
            with torch.no_grad():
                ema_out = self.teacher(batch_data)
            for m in self.teacher.modules():
                if isinstance(m, MyBatchNorm):
                    m.calibrate_mode = False
        # ema_out = self.teacher(batch_data)
        entropy = self.self_softmax_entropy(ema_out)
        stu_out = self.model(strong_aug)
        entropy_mask = (entropy < 0.2 * math.log(10))
        l_sup = (softmax_entropy(stu_out, ema_out) * 1.0).mean()
        # l_sup = (softmax_entropy(stu_out, ema_out) * 1.0)[entropy_mask].mean()
        # confidence threshold
        # entropy = softmax_entropy(outputs)
        # filter = torch.where(entropy < self.margin)[0]  
        # # ratio = len(filter) / batch_size if batch_size > 0 else 0.0
        # # print(f"filter_ids_1占内存样本比例: {ratio:.4f} ({len(filter)}/{batch_size})")
        # # logits = outputs[filter]
        # entropy = entropy[filter]
        # # pseudo_label = pseudo_label [filter]
        # loss = entropy.mean(0)
        # print("entropy_loss:", loss)
        if self.lambda_bn_w > 0:

            l_soft_alignment = []
            for m in self.model.modules():
                if isinstance(m, MyBatchNorm):
                    l_soft_alignment.append(m.get_soft_alignment_loss_weight())
            l_soft_alignment = torch.stack(l_soft_alignment).sum()
            l_soft_alignment = l_soft_alignment * self.lambda_bn_w
        else:
            l_soft_alignment = torch.tensor(0.0).cuda()
        # print("l_soft_alignment:", l_soft_alignment)
        # loss += l_soft_alignment
        loss = l_sup + l_soft_alignment 
        # self.ce_loss = torch.nn.CrossEntropyLoss()
        # loss_pl = 1.0 * self.ce_loss(outputs, pseudo_label) 
        # print("loss_pl:", loss_pl)
        # loss += loss_pl
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # for m in self.model.modules():  
        #     if isinstance(m, MyBatchNorm):
        #         m.regularize_statistics()
        # for m in self.teacher.modules():  
        #     if isinstance(m, MyBatchNorm):
        #         m.regularize_statistics()
        self.update_teacher()
        outputs = teacher_outputs
        return outputs
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
    @torch.no_grad()
    def update_teacher(self):
        for t_params, s_params in zip(self.teacher.parameters(), self.model.parameters()):
            t_params.data.mul_(self.ema_decay).add_(s_params.data * (1 - self.ema_decay))

# @torch.jit.script
# def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(1) * x.log_softmax(1)).sum(1)
@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
