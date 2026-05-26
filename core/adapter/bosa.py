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

# @torch.jit.script
def grad_refill(grad_out, removed_idxs, remained_idxs, full_weight, full_bias, running_var, eps,
                weight_grad, bias_grad, detached_x_grad):
    """
    grad_out: (N, C, H, W)
    full_weight/bias/running_var: (1, C, 1, 1)
    """

    # ---- normalize BN params to (C,) ----
    w = full_weight.view(-1)
    b = full_bias.view(-1)
    rv = running_var.view(-1)

    # ---- removed channel gradients ----
    grad_rm = grad_out[:, removed_idxs, :, :]
    grad_rm_sum = grad_rm.sum(dim=(0, 2, 3), keepdim=True)  # (1, C_rm, 1, 1)

    # ---- grad_x ----
    if detached_x_grad is None:
        grad_x = None
    else:
        grad_x = torch.zeros_like(grad_out)

        N, _, H, W = grad_out.shape
        ch_n = N * H * W

        grad_x_factor = (w[removed_idxs] / torch.sqrt(rv[removed_idxs] + eps)).view(1, -1, 1, 1)
        
        grad_x_rm = - grad_rm_sum / ch_n * grad_x_factor + grad_rm * grad_x_factor
        
        grad_x.index_copy_(1, removed_idxs, grad_x_rm)
        grad_x.index_copy_(1, remained_idxs, detached_x_grad)

    # ---- grad_w ----
    if full_weight is not None and weight_grad is not None:
        grad_w = torch.zeros_like(w)
        grad_w.index_copy_(0, remained_idxs, weight_grad.view(-1))
        grad_w = grad_w.view_as(full_weight)
    else:
        grad_w = None

    # ---- grad_b ----
    if full_bias is not None and bias_grad is not None:
        grad_b = torch.zeros_like(b)
        grad_b.index_copy_(0, remained_idxs, bias_grad.view(-1))
        # grad_rm_sum must be flattened here
        grad_b.index_copy_(0, removed_idxs, grad_rm_sum.view(-1))
        grad_b = grad_b.view_as(full_bias)
    else:
        grad_b = None

    return grad_x, grad_w, grad_b

class StochCacheFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, MyBatchNorm, preserve_rng_state, x, weight, bias):
        if any(t.requires_grad for t in (x, weight, bias)):
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
                    weight_abs = weight.abs().view(-1) 
                    sorted_indices = torch.argsort(weight_abs, dim=0)
                    ctx.removed_idxs = sorted_indices[:n_rm]  
                    ctx.remained_idxs = sorted_indices[n_rm:]  
                    ctx.n_channels = n_channels
                # if n_rm > 0:    
                #     idxs = torch.randperm(n_channels, device=x.device)
                    
                #     ctx.removed_idxs = idxs[:n_rm]
                #     ctx.remained_idxs = idxs[n_rm:]
                #     ctx.n_channels = n_channels

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
            weight, bias = MyBatchNorm.weight.detach(), MyBatchNorm.bias.detach()
            running_mean, running_var = MyBatchNorm.mu.detach(), MyBatchNorm.sigma.detach()
            
            # Use separate variables for sliced versions to avoid overwriting full parameters
            rm_s, rv_s = running_mean, running_var
            w_s, b_s = weight, bias

            if ctx.n_rm > 0:
                # Get remained sliced params
                with torch.no_grad():
                    w_s = MyBatchNorm.weight[:, ctx.remained_idxs, :, :]
                    b_s = MyBatchNorm.bias[:, ctx.remained_idxs, :, :]
                    rm_s = running_mean[:, ctx.remained_idxs, :, :]
                    rv_s = running_var[:, ctx.remained_idxs, :, :]
            # requires grad
            detached_x.requires_grad = x.requires_grad
            w_s.requires_grad, b_s.requires_grad = MyBatchNorm.weight.requires_grad, MyBatchNorm.bias.requires_grad
            with torch.enable_grad(), \
                torch.amp.autocast('cuda', **ctx.gpu_autocast_kwargs), \
                torch.amp.autocast('cpu', **ctx.cpu_autocast_kwargs):
                if not ctx.req_cache:
                    w_s = w_s.detach()
                    b_s = b_s.detach()
                y = batch_norm(rm_s, rv_s, detached_x, w_s, b_s, eps=MyBatchNorm.eps)
            with torch.no_grad():
                if ctx.n_rm > 0:
                    remained_idxs = ctx.remained_idxs
                    removed_idxs = ctx.removed_idxs

                    if torch.is_tensor(y) and y.requires_grad:
                        torch.autograd.backward([y], [grad_out[:, remained_idxs, :, :]])

                    # Use full running_var here
                    grad_x, grad_w, grad_b = grad_refill(grad_out, removed_idxs, remained_idxs, MyBatchNorm.weight, MyBatchNorm.bias, running_var, MyBatchNorm.eps, w_s.grad, b_s.grad, detached_x.grad)
                else:
                    if torch.is_tensor(y) and y.requires_grad:
                        torch.autograd.backward([y], [grad_out])


                    grad_x = detached_x.grad
                    grad_w = weight.grad 
                    grad_b = bias.grad 

        return None, None, grad_x, grad_w, grad_b
    
class MyBatchNorm(nn.Module):
    # Static variables for on-the-fly dynamic sparsity
    global_memory_sum = 0.0
    current_batch_max_score = 1e-8
    previous_batch_max_score = 1e-8

    def __init__(self, bn_init: nn.BatchNorm2d):
        super().__init__()
        
        self.register_buffer("running_mean", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("running_var", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("mu", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("sigma", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
        self.weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1))
        self.bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1))

        self.eps = bn_init.eps
        self.prune_q = 0.0
        self.full_matched = False   
        
        # New attributes for adaptive computing
        self.is_teacher = False
        self.activation_size = 0
        self.kl_score = 0.0

    def reset_statistic(self):
        self.full_matched = False

    @classmethod
    def reset_runtime_state(cls):
        cls.global_memory_sum = 0.0
        cls.current_batch_max_score = 1e-8
        cls.previous_batch_max_score = 1e-8
    
    def forward_w_update_stats(self, X):
        if self.training:
            buffer_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
            buffer_var = torch.var(X, dim=(0, 2, 3), keepdim=True, unbiased=True).clone()
            
            dist = gauss_symm_kl_divergence(buffer_mean, buffer_var, self.mu, self.sigma, eps=self.eps)
            self.kl_score = dist.mean().item()
            
            alpha = 1. - torch.exp(torch.tensor(- 0.1 * self.kl_score, device=self.mu.device)).item()
            self.full_matched = False

            if alpha > 0.0:
                self.full_matched = False
                self.weight.requires_grad, self.bias.requires_grad = True, True
            else:
                self.full_matched = True
                self.weight.requires_grad, self.bias.requires_grad = False, False

            self.mu.data = alpha * buffer_mean + (1 - alpha) * self.mu.data.clone()
            self.sigma.data = alpha * buffer_var + (1 - alpha) * self.sigma.data.clone()

            # cifar10 1.1 0.2 cifar100 0.1 0.1 imagenet 10 0.03 
            gradient_mean = 2 * (self.mu - self.running_mean)

            target_std = torch.sqrt(self.sigma + self.eps)
            source_std = torch.sqrt(self.running_var + self.eps)
            gradient_std = 2 * target_std - 2 * source_std
            target_std = target_std - alpha * gradient_std

            self.mu.copy_(self.mu - alpha * gradient_mean)
            self.sigma.copy_(target_std ** 2)

    def forward(self, X):
        # 1. Update memory tracking (only main model, not teacher)
        mem = X.numel()
        if self.activation_size == 0 and not self.is_teacher:
            self.activation_size = mem
            MyBatchNorm.global_memory_sum += mem

        # 2. Update stats and get current KL divergence
        self.forward_w_update_stats(X)
        
        # 3. Dynamic Pruning Ratio Calculation (On-the-fly)
        if self.training and not self.is_teacher:
            if MyBatchNorm.global_memory_sum > 0 and self.activation_size > 0:
                score = self.kl_score * math.log(MyBatchNorm.global_memory_sum / self.activation_size)
                # Keep tracking max score for the next batch normalization
                MyBatchNorm.current_batch_max_score = max(MyBatchNorm.current_batch_max_score, score)

                # Use previous batch max score to normalize on-the-fly
                if MyBatchNorm.previous_batch_max_score > 1e-8:
                    importance = score / MyBatchNorm.previous_batch_max_score
                    importance = min(1.0, max(0.0, importance))
                    self.prune_q = 1.0 - importance
                else:
                    self.prune_q = 0.0
            else:
                self.prune_q = 0.0

        return StochCacheFunction.apply(self, True, X, self.weight, self.bias)

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
