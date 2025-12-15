import torch
import torch.nn as nn
from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states, detach_variable, checkpoint
from ..utils import memory
from .base_adapter import BaseAdapter
import torch.nn.functional as F
import os
import math
import copy
import torchvision.transforms as transforms
from . import my_transforms
import PIL

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

# @torch.jit.script  # 若需jit编译，需确保所有操作兼容jit，暂时注释便于调试
def grad_refill(grad_out, removed_idxs, remained_idxs, full_weight, full_bias, running_var, eps,
                weight_grad, bias_grad, detached_x_grad):
    if detached_x_grad is None:
        grad_x = None
    else:
        grad_x = torch.zeros_like(grad_out)  # (N, C_ori, H, W)
        grad_rm = grad_out[:, removed_idxs, :, :]
        grad_rm_sum = torch.sum(grad_rm, dim=(0, 2, 3), keepdim=True)  # (1, n_rm, 1, 1)
        
        ch_n = grad_out.shape[0] * grad_out.shape[2] * grad_out.shape[3]
        ch_n = ch_n if ch_n != 0 else 1  
        
        full_weight_rm = full_weight[:, removed_idxs, :, :]  # 4D切片，维度1是通道
        running_var_rm = running_var[:, removed_idxs, :, :]
        
        grad_x_factor = full_weight_rm / torch.sqrt(running_var_rm + eps)  # (1, n_rm, 1, 1)
        
        grad_rm_filled = (- grad_rm_sum / ch_n + grad_rm) * grad_x_factor
        
        grad_x.index_copy_(
            dim=1,  
            index=removed_idxs,
            source=grad_rm_filled
        )
    
        grad_x.index_copy_(
            dim=1,
            index=remained_idxs,
            source=detached_x_grad
        )

    if full_weight is None or weight_grad is None:
        grad_w = None
    else:
        grad_w = torch.zeros_like(full_weight)  # (1, C_ori, 1, 1)
        weight_grad_4d = weight_grad.view(1, -1, 1, 1) if weight_grad.dim() != 4 else weight_grad
        grad_w.index_copy_(
            dim=1,  # 4D张量的通道维度是1
            index=remained_idxs,
            source=weight_grad_4d
        )

    if full_bias is None or bias_grad is None:
        grad_b = None
    else:
        grad_b = torch.zeros_like(full_bias)  
        bias_grad_4d = bias_grad.view(1, -1, 1, 1) if bias_grad.dim() != 4 else bias_grad
        grad_b.index_copy_(
            dim=1,
            index=remained_idxs,
            source=bias_grad_4d
        )
        
        if 'grad_rm_sum' in locals():  
            grad_rm_sum_squeezed = grad_rm_sum.view(1, -1, 1, 1)  # (1, n_rm, 1, 1)
            grad_b.index_copy_(
                dim=1,
                index=removed_idxs,
                source=grad_rm_sum_squeezed
            )
    return grad_x, grad_w, grad_b

class StochCacheFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, MyBatchNorm, preserve_rng_state, x, weight, bias):
        any_requires_grad = x.requires_grad or weight.requires_grad or bias.requires_grad
        if not any_requires_grad:
            with torch.no_grad():
                y = batch_norm(MyBatchNorm.mu, MyBatchNorm.sigma, x, weight, bias, eps=MyBatchNorm.eps)
            ctx.req_cache = False
            return y
        check_backward_validity([x, weight, bias])
        ctx.MyBatchNorm = MyBatchNorm
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(x)

        with torch.no_grad():
            y = batch_norm(MyBatchNorm.mu, MyBatchNorm.sigma, x, weight, bias, eps=MyBatchNorm.eps)
            req_cache = not MyBatchNorm.full_matched

            ctx.req_cache = req_cache
            if req_cache:  
                n_channels = x.size(1)
                n_rm = int(n_channels * MyBatchNorm.prune_q)
                ctx.n_rm = n_rm
                if n_rm > 0:
                    weight_abs = weight.abs().squeeze()  # 4D[1,C,1,1] → 1D[C]
                    sorted_indices = torch.argsort(weight_abs, dim=0)  
                    ctx.removed_idxs = sorted_indices[:n_rm]  
                    ctx.remained_idxs = sorted_indices[n_rm:]  
                    ctx.n_channels = n_channels
                    
                    x_slice = x.index_select(1, ctx.remained_idxs)
                    x_slice.requires_grad = x.requires_grad
                    x = x_slice

                ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        if not ctx.req_cache: 
            return None, None, grad_out, None, None
        x, = ctx.saved_tensors
        MyBatchNorm = ctx.MyBatchNorm
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

            detached_x = x.detach()
            weight, bias = MyBatchNorm.weight, MyBatchNorm.bias
            running_mean, running_var = MyBatchNorm.mu.detach(), MyBatchNorm.sigma.detach()
            
            if ctx.n_rm > 0:
                # Get remained sliced params
                with torch.no_grad():
                    weight = MyBatchNorm.weight[:, ctx.remained_idxs, :, :]  # 维度1是通道维
                    bias = MyBatchNorm.bias[:, ctx.remained_idxs, :, :]
                    running_mean = running_mean[:, ctx.remained_idxs, :, :]
                    running_var = running_var[:, ctx.remained_idxs, :, :]
            # requires grad
            detached_x.requires_grad = x.requires_grad
            weight.requires_grad, bias.requires_grad = MyBatchNorm.weight.requires_grad, MyBatchNorm.bias.requires_grad
            with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                if not ctx.req_cache:
                    weight = weight.detach()
                    bias = bias.detach()
                y = batch_norm(running_mean, running_var, detached_x, weight, bias, eps=MyBatchNorm.eps)
            with torch.no_grad():
                if ctx.n_rm > 0:
                    remained_idxs = ctx.remained_idxs
                    removed_idxs = ctx.removed_idxs

                    if torch.is_tensor(y) and y.requires_grad:
                        torch.autograd.backward([y], [grad_out[:, remained_idxs, :, :]])

                    grad_x, grad_w, grad_b = grad_refill(grad_out, removed_idxs, remained_idxs, MyBatchNorm.weight, MyBatchNorm.bias, MyBatchNorm.running_var, MyBatchNorm.eps, weight.grad, bias.grad, detached_x.grad)
                else:
                    grad_x = detached_x.grad
                    grad_w = weight.grad 
                    grad_b = bias.grad 

        return None, None, grad_x, grad_w, grad_b

class MyBatchNorm(nn.Module):
    def __init__(self, bn_init: nn.BatchNorm2d, prune_q):
        super().__init__()
        
        self.register_buffer("running_mean", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("running_var", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("mu", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("sigma", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
        self.source_weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1))
        self.source_bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1))
        self.weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1))
        self.bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1))

        self.eps = 1e-5
        self.prune_q = prune_q
        self.full_matched = False   

    def reset_statistic(self):
        # self.full_matched = True
        # self.alpha = datta_alpha
        self.reset_mean = True

    def forward_w_update_stats(self, X):
        if self.training:
            buffer_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
            buffer_var = torch.var(X, dim=(0, 2, 3), keepdim=True, unbiased=True).clone()
            
            dist = gauss_symm_kl_divergence(buffer_mean, buffer_var, self.mu, self.sigma, eps=self.eps)
            self.alpha = 1. - torch.exp(- 0.1 * dist.mean()).item()

            if self.alpha > 0.0:
                self.full_matched = False
                self.weight.requires_grad, self.bias.requires_grad = True, True
            else:
                self.full_matched = True
                self.weight.requires_grad, self.bias.requires_grad = False, False

            self.mu.data = self.alpha * buffer_mean + (1 - self.alpha) * self.mu.data.clone()
            self.sigma.data = self.alpha * buffer_var + (1 - self.alpha) * self.sigma.data.clone()

            self.alpha = 1. - torch.exp(- 0.1 * dist.mean()).item()
            gradient_mean = 2 * (self.mu - self.running_mean)

            target_std = torch.sqrt(self.sigma + self.eps)
            source_std = torch.sqrt(self.running_var + self.eps)
            gradient_std = 2 * target_std - 2 * source_std
            target_std = target_std - self.alpha * gradient_std

            self.mu.copy_(self.mu - self.alpha * gradient_mean)
            self.sigma.copy_(target_std ** 2)

    def forward(self, X):
        self.forward_w_update_stats(X)
        return StochCacheFunction.apply(self, True, X, self.weight, self.bias)

class MERT(BaseAdapter):
    def __init__(self, cfg, model, optimizer):  
        self.prune_q = cfg.ADAPTER.MERT.PRUNE_Q
        self.l1_lambda = cfg.ADAPTER.MERT.L1_LAMBDA 
        self.ema_decay = cfg.ADAPTER.MERT.EMA_DECAY
        super(MERT, self).__init__(cfg, model, optimizer)
        self.mem = memory.CSTU(capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, num_class=cfg.CORRUPTION.NUM_CLASS, lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U)
        self.teacher = copy.deepcopy(self.model)
        self.transform = get_tta_transforms(dataset='cifar')
        for p in self.teacher.parameters():
            p.requires_grad = False
            p.detach_()
        return

    def forward_and_adapt(self, batch_data):
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

        self.update_model()
        return teacher_outputs

    def update_model(self):
        self.model.train()
        self.teacher.train()
        imgs, ages = self.mem.get_memory()
        imgs = torch.stack(imgs)
        strong_aug = self.transform(imgs)
        
        ema_out = self.teacher(imgs)
        stu_out = self.model(strong_aug)
        entropy = self.self_softmax_entropy(ema_out)
        
        loss = (softmax_entropy(stu_out, ema_out)).mean()
        bn_l1_loss = compute_bn_weight_l1_loss(self.model, self.l1_lambda)
        loss += bn_l1_loss
        
        # if has_accum_bn_grad(self.model):
        #     loss.backward()
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()
        self.update_teacher()

    @torch.no_grad()
    def update_teacher(self):
        for t_params, s_params in zip(self.teacher.parameters(), self.model.parameters()):
            t_params.data.mul_(self.ema_decay).add_(s_params.data * (1 - self.ema_decay))

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
        model = self.replace_bn_with_custom(model, lambda m: MyBatchNorm(m, prune_q = self.prune_q))
        model.requires_grad_(False)
        for m in model.modules():  
            if isinstance(m, MyBatchNorm):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        return model

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

def compute_bn_weight_l1_loss(model, l1_lambda=1e-5):
    l1_loss = 0.0
    for m in model.modules():
        if isinstance(m, MyBatchNorm):
            # 提取weight并计算L1范数（4D→1D，避免维度影响）
            weight = m.weight.squeeze()  # [1,C,1,1] → [C]
            l1_loss += torch.norm(weight, p=1)  # L1范数：绝对值之和
    l1_loss = l1_lambda * l1_loss
    return l1_loss

@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def has_accum_bn_grad(model):
    """Return True, if at least one param has grad."""
    all_mached = True
    has_acc_bn = False
    for m in model.modules():
        if isinstance(m, MyBatchNorm):
            has_acc_bn = True
            if not m.full_matched:
                all_mached = False
                break
    if has_acc_bn and all_mached:
        return False
    return True