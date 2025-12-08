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
def compute_bn_weight_l1_loss(model, l1_lambda=1e-5):
    """
    计算所有MyBatchNorm层weight的L1 Loss
    参数：
        model: 模型实例
        l1_lambda: L1正则化系数（可调节，控制L1 Loss的权重）
    返回：
        l1_loss: 所有MyBatchNorm层weight的L1 Loss
    """
    l1_loss = 0.0
    for m in model.modules():
        if isinstance(m, MyBatchNorm):
            # 提取weight并计算L1范数（4D→1D，避免维度影响）
            weight = m.weight.squeeze()  # [1,C,1,1] → [C]
            l1_loss += torch.norm(weight, p=1)  # L1范数：绝对值之和
    # 乘以正则化系数
    l1_loss = l1_lambda * l1_loss
    return l1_loss
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
def batch_norm(mean, var, X, weight, bias, eps):

    X_hat = (X - mean) / torch.sqrt(var + eps)

    Y = weight * X_hat + bias  # Scale and shift

    return Y

class StochCacheFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, MyBatchNorm, preserve_rng_state, x, weight, bias):
        check_backward_validity([x, weight, bias])
        ctx.MyBatchNorm = MyBatchNorm
        ctx.preserve_rng_state = preserve_rng_state
        
        # 保存autocast状态
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }
        ctx.cpu_autocast_kwargs = {
            "enabled": torch.is_autocast_cpu_enabled(),
            "dtype": torch.get_autocast_cpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }
        
        # 保存随机数状态
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_cuda_in_fwd = torch.cuda._initialized
            if ctx.had_cuda_in_fwd:
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(x)
        else:
            ctx.had_cuda_in_fwd = False
        
        # 前向计算BN
        with torch.no_grad():
            y = batch_norm(MyBatchNorm.mu, MyBatchNorm.sigma, x, weight, bias, eps=MyBatchNorm.eps)
            req_cache = not MyBatchNorm.full_matched
            ctx.req_cache = req_cache
            
            # 剪枝逻辑：按weight绝对值选通道
            if req_cache:
                n_channels = x.size(1)
                n_rm = int(n_channels * MyBatchNorm.prune_q)
                ctx.n_rm = n_rm
                ctx.n_channels = n_channels
                
                if n_rm > 0:
                    # 提取weight绝对值并排序（通道维度）
                    weight_abs = weight.abs().squeeze()  # [1,C,1,1] → [C]
                    sorted_indices = torch.argsort(weight_abs, dim=0)  # 升序
                    ctx.removed_idxs = sorted_indices[:n_rm]  # 剪枝通道（绝对值最小）
                    ctx.remained_idxs = sorted_indices[n_rm:]  # 保留通道
                    
                    # 切片输入并保存
                    x_slice = x.index_select(1, ctx.remained_idxs)
                    x_slice.requires_grad = x.requires_grad
                    ctx.save_for_backward(x_slice)
                else:
                    ctx.save_for_backward(x)
            else:
                ctx.n_rm = 0
                ctx.save_for_backward(x)
        
        return y
    @staticmethod
    def backward(ctx, grad_out):
        """修复反向传播逻辑：正确计算并回填参数梯度"""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an inputs parameter "
                "is passed to .backward(). Please use .backward() without inputs argument."
            )
        
        # 无需缓存时直接返回梯度
        if not ctx.req_cache or ctx.n_rm == 0:
            return None, None, grad_out, torch.zeros_like(ctx.MyBatchNorm.weight), torch.zeros_like(ctx.MyBatchNorm.bias)
        
        # 恢复随机数状态
        x_sliced, = ctx.saved_tensors
        MyBatchNorm = ctx.MyBatchNorm
        rng_devices = ctx.fwd_gpu_devices if (ctx.preserve_rng_state and ctx.had_cuda_in_fwd) else []
        
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
        
        # 准备反向输入（保留梯度链路）
        detached_x = x_sliced.detach()
        detached_x.requires_grad = True  # 强制开启梯度
        remained_idxs = ctx.remained_idxs
        removed_idxs = ctx.removed_idxs
        
        # 切片参数（不detach，保留梯度链路）
        weight_sliced = MyBatchNorm.weight[:, remained_idxs, :, :]
        bias_sliced = MyBatchNorm.bias[:, remained_idxs, :, :]
        mu_sliced = MyBatchNorm.mu[:, remained_idxs, :, :]
        sigma_sliced = MyBatchNorm.sigma[:, remained_idxs, :, :]
        
        # 强制开启参数梯度
        weight_sliced.requires_grad = True
        bias_sliced.requires_grad = True
        
        # 重新前向计算（带梯度）
        with torch.enable_grad(), \
             torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
             torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            y = batch_norm(mu_sliced, sigma_sliced, detached_x, weight_sliced, bias_sliced, eps=MyBatchNorm.eps)
        
        # 反向传播计算切片梯度
        grad_out_sliced = grad_out[:, remained_idxs, :, :]
        torch.autograd.backward([y], [grad_out_sliced])
        
        # 回填梯度到完整维度
        grad_x, grad_weight, grad_bias = grad_refill(
            grad_out, removed_idxs, remained_idxs,
            MyBatchNorm.weight, MyBatchNorm.bias, MyBatchNorm.sigma, MyBatchNorm.eps,
            weight_sliced.grad, bias_sliced.grad, detached_x.grad
        )
        
        # 返回值顺序：ctx的forward参数对应的梯度（MyBatchNorm, preserve_rng_state, x, weight, bias）
        # 注意：非张量参数返回None，张量参数返回对应梯度
        return None, None, grad_x, grad_weight, grad_bias

class MyBatchNorm(nn.Module):
    _global_bn_counter = 0  # 静态计数器，确保每层有唯一 ID
    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.5, log_dir="/root/WZR/TRIBE/statistic"):
        super().__init__()
        
        try:
            num_features = bn_init.num_features
        except AttributeError:
            num_features = bn_init.num_feature

        self.register_buffer("running_mean", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("running_var", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
        self.source_weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1))
        self.source_bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("mu", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("sigma", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
        self.weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1))
        self.bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1))

        self.eps = 1e-5
        self.prune_q = 0.0
        

        self.calibrate_mode = False
        self.alpha = datta_alpha
        self.full_matched = False
        self.reset_mean = True
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 分配唯一 ID
        MyBatchNorm._global_bn_counter += 1
        self.layer_id = MyBatchNorm._global_bn_counter

        # 日志文件路径
        self.log_path = os.path.join(log_dir, f"bn_layer_{self.layer_id:03d}.txt")

        # 初始化日志文件
        with open(self.log_path, "w") as f:
            f.write(f"=== BN Layer {self.layer_id} Logging Start ===\n")
            f.write(f"[Init BN] running_mean: {bn_init.running_mean.detach().cpu().numpy().tolist()}\n")
            f.write(f"[Init BN] running_var:  {bn_init.running_var.detach().cpu().numpy().tolist()}\n")

    def reset_statistic(self):
        # self.full_matched = True
        # self.alpha = datta_alpha
        self.reset_mean = True

    def set_alpha(self, alpha):
        self.reset_mean = True
        self.alpha = alpha
    def get_soft_alignment_loss_weight(self):
        # return F.mse_loss(self.weight, self.source_weight) + F.mse_loss(self.bias, self.source_bias)
        return torch.sum((self.weight - self.source_weight) ** 2) + torch.sum((self.bias - self.source_bias) ** 2)
    def forward_w_update_stats(self, X):
        if self.reset_mean:
            self.reset_mean = False
            # self.weight.requires_grad, self.bias.requires_grad = False, False
        else:
            batch_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
            batch_var = torch.mean((X - batch_mean) ** 2, dim=(0, 2, 3), keepdim=True).clone()
            
            dist = gauss_symm_kl_divergence(
                batch_mean, batch_var, self.mu, self.sigma, eps=self.eps)
            adaptive_alpha = 1. - torch.exp(- 1.0 * dist.mean())
            self.alpha = adaptive_alpha.item()
            if self.alpha > 0.0:
                self.full_matched = False
                # self.weight.requires_grad, self.bias.requires_grad = True, True
            else:
                self.full_matched = True
                # self.weight.requires_grad, self.bias.requires_grad = False, False
            self.mu.data = self.alpha * batch_mean + (1 - self.alpha) * self.mu.data.clone()
            self.sigma.data = self.alpha * batch_var + (1 - self.alpha) * self.sigma.data.clone()

            adaptive_alpha = 1. - torch.exp(- 0.1 * dist.mean())
            self.alpha = adaptive_alpha.item()
            gradient_mean = 2 * (self.mu - self.running_mean)
            target_std = torch.sqrt(self.sigma + self.eps)
            source_std = torch.sqrt(self.running_var + self.eps)
            gradient_std = 2 * target_std - 2 * source_std
            target_std = target_std - self.alpha * gradient_std

            self.mu.copy_(self.mu - self.alpha * gradient_mean)
            self.sigma.copy_(target_std ** 2)
    def forward(self, X):
        # self.forward_cache_size = list(X.shape)
        # self.forward_cache_size[1] = int(
        #     (1.-self.prune_q) * self.forward_cache_size[1])
        self.forward_w_update_stats(X)
        return StochCacheFunction.apply(self, True, X, self.weight, self.bias)

class BN(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        self.alpha = cfg.ADAPTER.BN.ALPHA  
        self.theta = cfg.ADAPTER.BN.THETA 
        super(BN, self).__init__(cfg, model, optimizer)
        self.has_calibrate = False
        self.lambda_bn_w = 0.0
        self.mem = memory.CSTU(capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, num_class=cfg.CORRUPTION.NUM_CLASS, lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U)
        self.ema_decay = 0.999
        self.teacher = copy.deepcopy(self.model)
        self.transform = get_tta_transforms(dataset='cifar')
        for p in self.teacher.parameters():
            p.requires_grad = False
            p.detach_()
        return

    def forward_and_adapt(self, batch_data):
        self.reset()
        with torch.no_grad():
            teacher_outputs = self.teacher(batch_data)
            teacher_probs = torch.softmax(teacher_outputs, dim=1)
            pseudo_label = torch.argmax(teacher_probs, dim=1)
            entropy = torch.sum(- teacher_probs * torch.log(teacher_probs + 1e-6), dim=1)
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
        self.update_model()
        
        return teacher_outputs
    def update_model(self):
        imgs, ages = self.mem.get_memory()
        imgs = torch.stack(imgs)
        strong_aug = self.transform(imgs)
        
        ema_out = self.teacher(imgs)
        stu_out = self.model(strong_aug)
        loss = (softmax_entropy(stu_out, ema_out) * 1.0).mean()
        bn_l1_loss = compute_bn_weight_l1_loss(self.model)
        print(f"bn_l1_loss: {bn_l1_loss}")
        loss += bn_l1_loss * 1
        if self.lambda_bn_w > 0:
            l_soft_alignment = []
            print(self.lambda_bn_w)
            for m in self.model.modules():
                if isinstance(m, MyBatchNorm):
                    l_soft_alignment.append(m.get_soft_alignment_loss_weight())
                    print(l_soft_alignment)
            l_soft_alignment = torch.stack(l_soft_alignment).sum()
            l_soft_alignment = l_soft_alignment * self.lambda_bn_w
        else:
            l_soft_alignment = torch.tensor(0.0).cuda()
        loss += l_soft_alignment
        if has_accum_bn_grad(self.model):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.update_teacher()
    @torch.no_grad()
    def update_teacher(self):
        for t_params, s_params in zip(self.teacher.parameters(), self.model.parameters()):
            t_params.data.mul_(self.ema_decay).add_(s_params.data * (1 - self.ema_decay))
    @staticmethod
    def self_softmax_entropy(x):
        return -(x.softmax(dim=-1) * x.log_softmax(dim=-1)).sum(dim=-1)
    def reset(self):
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()
        for m in self.teacher.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()
        self.has_calibrate = False

    def replace_bn_with_custom(self, model: nn.Module, custom_bn):
        for name, module in model.named_children():
            if isinstance(module, (nn.BatchNorm2d)):
                setattr(model, name, custom_bn(module))
            else:
                self.replace_bn_with_custom(module, custom_bn)
        return model

    def configure_model(self, model: nn.Module):
        model = self.replace_bn_with_custom(model, lambda m: MyBatchNorm(m, datta_alpha = self.alpha, log_dir = "/root/WZR/TRIBE/statistic"))
        model.requires_grad_(False)
        for m in model.modules():  # 25, 31 ,53
            if isinstance(m, MyBatchNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
        return model
        
    def calibrate_with_buffer(self):
        imgs, ages = self.mem.get_memory()
        if len(imgs) == 0:
            return  # 无缓存数据时跳过
        
        imgs = torch.stack(imgs)
        outputs = self.model(imgs) 
        print("outputs.requires_grad:", outputs.requires_grad)  # 应输出 True
        print("outputs.grad_fn:", outputs.grad_fn)  # 应不为 None
        confidence = torch.softmax(outputs, dim=1)
        outputs_above_threshold = []
        for j in range(confidence.shape[0]):
            if torch.max(confidence[j]) > self.theta:
                outputs_above_threshold.append(outputs[j])
        
        if len(outputs_above_threshold) != 0:
            outputs_above_threshold = torch.stack(outputs_above_threshold)
            loss = softmax_entropy(outputs_above_threshold).mean(0)
            
            if has_accum_bn_grad(self.model):
                self.optimizer.zero_grad()  # 先清零梯度
                loss.backward()
                self.optimizer.step()
@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
def gauss_kl_divergence(mean1, var1, mean2, var2, eps):
    # /// v1: relative to distribution 2 ///
    d1 = (torch.log(var2 + eps) - torch.log(var1 + eps))/2. + \
        (var1 + eps + (mean1 - mean2)**2) / 2. / (var2 + eps) - 0.5
    return d1
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
# @torch.jit.script  # 若需jit编译，需确保所有操作兼容jit，暂时注释便于调试
def grad_refill(grad_out, removed_idxs, remained_idxs, full_weight, full_bias, running_var, eps,
                weight_grad, bias_grad, detached_x_grad):
    """
    适配4D张量的梯度恢复函数（补全裁剪通道的梯度）
    输入张量维度规范：
    - grad_out: (N, C_ori, H, W)  原始输出梯度
    - full_weight/full_bias: (1, C_ori, 1, 1)  4D的完整权重/偏置
    - running_var: (1, C_ori, 1, 1)  4D的运行方差
    - weight_grad/bias_grad: (1, C_rem, 1, 1)  裁剪后权重/偏置梯度
    - detached_x_grad: (N, C_rem, H, W)  裁剪后输入梯度
    - removed_idxs/remained_idxs: 1D张量 裁剪/保留的通道索引
    """
    
    # ========== 1. 处理输入梯度 grad_x ==========
    if detached_x_grad is None:
        grad_x = None
    else:
        grad_x = torch.zeros_like(grad_out)  # (N, C_ori, H, W)
        # 提取裁剪通道的输出梯度 (N, n_rm, H, W)
        grad_rm = grad_out[:, removed_idxs, :, :]
        # 计算裁剪通道梯度的求和（保持维度：(1, n_rm, 1, 1)）
        grad_rm_sum = torch.sum(grad_rm, dim=(0, 2, 3), keepdim=True)  # (1, n_rm, 1, 1)
        
        # 计算通道数相关因子（避免除0）
        ch_n = grad_out.shape[0] * grad_out.shape[2] * grad_out.shape[3]
        ch_n = ch_n if ch_n != 0 else 1  # 兜底
        
        # 提取裁剪通道的权重和运行方差（4D切片：(1, n_rm, 1, 1)）
        full_weight_rm = full_weight[:, removed_idxs, :, :]  # 4D切片，维度1是通道
        running_var_rm = running_var[:, removed_idxs, :, :]
        
        # 计算梯度填充因子（保持4D，避免维度展平）
        grad_x_factor = full_weight_rm / torch.sqrt(running_var_rm + eps)  # (1, n_rm, 1, 1)
        
        # 计算裁剪通道的梯度值（维度匹配：(N, n_rm, H, W)）
        grad_rm_filled = (- grad_rm_sum / ch_n + grad_rm) * grad_x_factor
        
        # 填充裁剪通道梯度（维度1）
        grad_x.index_copy_(
            dim=1,  # 明确指定通道维度
            index=removed_idxs,
            source=grad_rm_filled
        )
        # 填充保留通道梯度（维度1）
        grad_x.index_copy_(
            dim=1,
            index=remained_idxs,
            source=detached_x_grad
        )

    # ========== 2. 处理权重梯度 grad_w ==========
    if full_weight is None or weight_grad is None:
        grad_w = None
    else:
        grad_w = torch.zeros_like(full_weight)  # (1, C_ori, 1, 1)
        # 填充保留通道的权重梯度（维度1）
        # 确保weight_grad是4D：若为2D/3D则reshape
        weight_grad_4d = weight_grad.view(1, -1, 1, 1) if weight_grad.dim() != 4 else weight_grad
        grad_w.index_copy_(
            dim=1,  # 4D张量的通道维度是1
            index=remained_idxs,
            source=weight_grad_4d
        )

    # ========== 3. 处理偏置梯度 grad_b ==========
    if full_bias is None or bias_grad is None:
        grad_b = None
    else:
        grad_b = torch.zeros_like(full_bias)  # (1, C_ori, 1, 1)
        # 填充保留通道的偏置梯度（维度1）
        bias_grad_4d = bias_grad.view(1, -1, 1, 1) if bias_grad.dim() != 4 else bias_grad
        grad_b.index_copy_(
            dim=1,
            index=remained_idxs,
            source=bias_grad_4d
        )
        # 填充裁剪通道的偏置梯度（来自grad_rm_sum，需匹配维度）
        if 'grad_rm_sum' in locals():  # 确保grad_rm_sum已定义
            grad_rm_sum_squeezed = grad_rm_sum.view(1, -1, 1, 1)  # (1, n_rm, 1, 1)
            grad_b.index_copy_(
                dim=1,
                index=removed_idxs,
                source=grad_rm_sum_squeezed
            )
    print("=== 梯度调试 ===")
    print(f"weight_grad 均值: {weight_grad.mean().item() if weight_grad is not None else 'None'}")
    print(f"bias_grad 均值: {bias_grad.mean().item() if bias_grad is not None else 'None'}")
    print(f"填充后grad_w 均值: {grad_w.mean().item() if grad_w is not None else 'None'}")
    print(f"填充后grad_b 均值: {grad_b.mean().item() if grad_b is not None else 'None'}")
    return grad_x, grad_w, grad_b