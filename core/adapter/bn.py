import torch
import torch.nn as nn
from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states, detach_variable, checkpoint
from ..utils import memory
from .base_adapter import BaseAdapter
import torch.nn.functional as F
import os
from torch.autograd import Function
# 当不设置剪枝和分层更新的话,这个代码对于cifar10c数据集准确率最高,能达到78%
def batch_norm(current_mean, current_var, x, weight, bias, eps):
    """BN enabling BP using given stats."""
    eps = torch.tensor([eps], dtype=current_var.dtype, device=current_var.device)
    _var = torch.sqrt(torch.maximum(current_var, eps)).view((1, -1, 1, 1)).detach()
    _mean = current_mean.view((1, -1, 1, 1))
    x_norm = (x - _mean) / _var
    # x_norm = (x - current_mean.view((1, -1, 1, 1))) / torch.sqrt(current_var + eps).view((1, -1, 1, 1))

    # re-norm refer to 'Online Normalization for Training Neural Networks'
    # x_norm = x_norm / torch.sqrt(torch.mean(x_norm**2, dim=(0,2,3), keepdim=True))
    # print(f"$$$ self.weight {self.weight.shape} self.bias {self.bias.shape}")
    if weight is not None and bias is not None:
        y = x_norm * weight.view((1, -1, 1, 1)) + bias.view((1, -1, 1, 1))
    else:
        y = x_norm
    return y

class StochCacheFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, MyBatchNorm, preserve_rng_state, x, weight, bias):
        check_backward_validity([x, weight, bias])
        # 检查输入是否满足用于反向重算的条件
        # print(f"### x req grad: {x.requires_grad}")
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
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(x)

        with torch.no_grad():
            # outputs, req_cache = run_function(x)
            y = batch_norm(MyBatchNorm.mu, MyBatchNorm.sigma, x, weight, bias, eps=MyBatchNorm.eps)
            req_cache = not MyBatchNorm.full_matched

            ctx.req_cache = req_cache
            if req_cache:  # require cache for norm grad or weight.
                # x = tensor_inputs[0]
                n_channels = x.size(1)
                n_rm = int(n_channels * MyBatchNorm.prune_q)
                ctx.n_rm = n_rm
                if n_rm > 0:
                    idxs = torch.randperm(n_channels, device=x.device)
                    
                    ctx.remained_idxs = idxs[n_rm:]
                    ctx.removed_idxs = idxs[:n_rm]
                    ctx.n_channels = n_channels
                    
                    # x_slice = x[:, ctx.remained_idxs, :, :]
                    x_slice = x.index_select(1, ctx.remained_idxs)
                    x_slice.requires_grad = x.requires_grad
                    x = x_slice

                ctx.save_for_backward(x)
        # else will not save tensor.
        return y

    @staticmethod
    def backward(ctx, grad_out):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        if not ctx.req_cache:  # no intermediate grad
            return None, None, grad_out, None, None
        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
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
            # detached_inputs = detach_variable((x,))

            detached_x = x.detach()
            # detach variables
            weight, bias = MyBatchNorm.weight.detach(), MyBatchNorm.bias.detach()
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

                    # Refill grad
                    if torch.is_tensor(y) and y.requires_grad:
                        torch.autograd.backward([y], [grad_out[:, remained_idxs, :, :]])

                    # if detached_x.grad is not None:  # this may ignore some grads on w/b,
                    grad_x, grad_w, grad_b = grad_refill(grad_out, removed_idxs, remained_idxs, MyBatchNorm.weight, MyBatchNorm.bias, MyBatchNorm.running_var, MyBatchNorm.eps, weight.grad, bias.grad, detached_x.grad)
                else:
                    # run backward() with only tensor that requires grad
                    if torch.is_tensor(y) and y.requires_grad:
                        torch.autograd.backward([y], [grad_out])

                    # retrive input grad
                    grad_x = detached_x.grad
                    grad_w = weight.grad 
                    grad_b = bias.grad 
                
                # grad_x = grad_out
                # grad_w = None
                # grad_b = None

        return None, None, grad_x, grad_w, grad_b
class MyBatchNorm(nn.Module):
    _global_bn_counter = 0  # 静态计数器，确保每层有唯一 ID
    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.5, log_dir="/root/WZR/TRIBE/statistic"):
        super().__init__()
        
        try:
            num_features = bn_init.num_features
        except AttributeError:
            num_features = bn_init.num_feature
        # 统一初始化为4D张量 [1, C, 1, 1]
        self.register_buffer("running_mean", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("running_var", bn_init.running_var.clone().detach().view(1, -1, 1, 1))
       
        self.weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1))
        self.bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1))

        self.eps = 1e-5
        self.prune_q = 0.1

        self.register_buffer("mu", bn_init.running_mean.clone().detach().view(1, -1, 1, 1))
        self.register_buffer("sigma", bn_init.running_var.clone().detach().view(1, -1, 1, 1))

        # self.register_buffer("mu", torch.ones(1, num_features, 1, 1))
        # self.register_buffer("sigma", torch.zeros(1, num_features, 1, 1))

        self.calibrate_mode = False
        self.alpha = datta_alpha
        self.full_matched = True
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
        self.reset_mean = True
        # self.weight.data = self.weight_init.clone().detach()
        # self.bias.data = self.bias_init.clone().detach()

    def set_calibrate_mode(self, mode: bool):
        """切换校准模式，并在结束校准时重置梯度开关"""
        if not mode and self.calibrate_mode:  # 从校准模式切换到非校准模式
            # 结束校准时，根据当前 alpha 重新设置梯度
            if self.alpha > 0.:
                self.full_matched = False
                self.weight.requires_grad = True
                self.bias.requires_grad = True
            else:
                self.full_matched = True
                self.weight.requires_grad = False
                self.bias.requires_grad = False
        self.calibrate_mode = mode  # 更新模式

    def set_alpha(self, alpha):
        self.reset_mean = True
        self.alpha = alpha

    def forward_w_update_stats(self, X):
        batch_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
        batch_var = torch.mean((X - batch_mean) ** 2, dim=(0, 2, 3), keepdim=True).clone()
        
        dist = gauss_symm_kl_divergence(
            batch_mean, batch_var, self.mu, self.sigma, eps=self.eps)
        adaptive_alpha = 1. - torch.exp(- 1.0 * dist.mean())
        self.alpha = adaptive_alpha.item()
        if self.reset_mean:
            self.reset_mean = False
            self.weight.requires_grad, self.bias.requires_grad = False, False
        else:
            if self.calibrate_mode:
                self.mu.data = self.alpha * batch_mean + (1 - self.alpha) * self.mu.data.clone()
                self.sigma.data = self.alpha * batch_var + (1 - self.alpha) * self.sigma.data.clone()
                self.weight.requires_grad, self.bias.requires_grad = False, False
            else:
                if self.alpha > 0.:
                    self.full_matched = False
                    self.weight.requires_grad, self.bias.requires_grad = True, True
                else:  # stop grad
                    self.full_matched = True
                    self.weight.requires_grad, self.bias.requires_grad = False, False
    def forward(self, X):
        # self.forward_cache_size = list(X.shape)
        # self.forward_cache_size[1] = int(
        #     (1.-self.prune_q) * self.forward_cache_size[1])
        self.forward_w_update_stats(X)
        return StochCacheFunction.apply(self, False, X, self.weight, self.bias)

class BN(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        self.alpha = cfg.ADAPTER.BN.ALPHA  
        self.theta = cfg.ADAPTER.BN.THETA 
        super(BN, self).__init__(cfg, model, optimizer)
        self.has_calibrate = False
        self.mem = memory.CSTU(capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, num_class=cfg.CORRUPTION.NUM_CLASS, lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U)

        return

    def forward_and_adapt(self, batch_data):
        self.reset()
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
        # 仅当第一次达到平衡时才更新统计量
        # if self.mem.is_balanced():
        #     self.has_calibrate = True
        # if not self.has_calibrate:
        #     self.calibrate_with_buffer()
        self.calibrate_with_buffer()
        outputs = self.model(batch_data)
        # confidence threshold
        confidence = torch.softmax(outputs, dim=1)
        outputs_above_threshold = []
        for j in range(confidence.shape[0]):
            if torch.max(confidence[j]) > self.theta:
                outputs_above_threshold.append(outputs[j])

        if len(outputs_above_threshold) != 0:
            outputs_above_threshold = torch.stack(outputs_above_threshold)
            loss = softmax_entropy(outputs_above_threshold).mean(0) 
            if has_accum_bn_grad(self.model):
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return outputs

    def reset(self):
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()
        self.has_calibrate = False
    def replace_bn_with_custom(self, model: nn.Module, custom_bn):
        for name, module in model.named_children():
            # if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
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
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.set_calibrate_mode(True)  # 进入校准模式
        
        imgs, ages = self.mem.get_memory()
        if len(imgs) > 0:
            imgs = torch.stack(imgs)
            with torch.no_grad():
                _ = self.model(imgs)  
        
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.set_calibrate_mode(False)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
# @torch.jit.script
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

    return grad_x, grad_w, grad_b