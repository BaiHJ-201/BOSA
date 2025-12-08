import torch
import torch.nn as nn
import os
import numpy as np

# 补充缺失的gauss_symm_kl_divergence函数
def gauss_symm_kl_divergence(mean1, var1, mean2, var2, eps=1e-5):
    var1 = var1 + eps
    var2 = var2 + eps
    kl = 0.5 * (torch.log(var2 / var1) + (var1 + (mean1 - mean2)**2) / var2 - 1)
    return kl

# 补充缺失的grad_refill函数
def grad_refill(grad_out, removed_idxs, remained_idxs, weight, bias, running_var, eps, 
                grad_weight_sliced, grad_bias_sliced, grad_x_sliced):
    """回填剪枝通道的梯度到完整维度"""
    # 初始化完整梯度
    grad_x = torch.zeros_like(grad_out)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    
    # 回填保留通道的梯度
    grad_x[:, remained_idxs, :, :] = grad_x_sliced
    grad_weight[:, remained_idxs, :, :] = grad_weight_sliced
    grad_bias[:, remained_idxs, :, :] = grad_bias_sliced
    
    # 剪枝通道梯度置0（或根据需求设置）
    grad_x[:, removed_idxs, :, :] = 0
    grad_weight[:, removed_idxs, :, :] = 0
    grad_bias[:, removed_idxs, :, :] = 0
    
    return grad_x, grad_weight, grad_bias

def check_backward_validity(tensors):
    """检查张量是否可用于反向传播"""
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise RuntimeError(f"Invalid tensor type: {type(t)}")
    return True

def get_device_states(x):
    """获取CUDA设备状态（兼容CPU）"""
    if torch.cuda.is_available() and x.is_cuda:
        devices = [x.device]
        states = [torch.cuda.get_rng_state(d) for d in devices]
        return devices, states
    return [], []

def set_device_states(devices, states):
    """恢复CUDA设备状态"""
    for d, s in zip(devices, states):
        torch.cuda.set_rng_state(s, device=d)

def batch_norm(mean, var, X, weight, bias, eps):
    X_hat = (X - mean) / torch.sqrt(var + eps)
    Y = weight * X_hat + bias  # Scale and shift
    return Y

class StochCacheFunction(torch.autograd.Function):
    """Stochastically cache data for backwarding (修复梯度传递逻辑)"""
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
    _global_bn_counter = 0  # 静态计数器
    
    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.5, log_dir="./bn_logs"):
        super().__init__()
        # 获取原始BN参数
        num_features = bn_init.num_features
        self.num_features = num_features
        
        # 注册可训练参数（核心：保留requires_grad=True）
        self.running_mean = nn.Parameter(bn_init.running_mean.clone().detach().view(1, -1, 1, 1), requires_grad=False)
        self.running_var = nn.Parameter(bn_init.running_var.clone().detach().view(1, -1, 1, 1), requires_grad=False)
        self.mu = nn.Parameter(bn_init.running_mean.clone().detach().view(1, -1, 1, 1), requires_grad=False)
        self.sigma = nn.Parameter(bn_init.running_var.clone().detach().view(1, -1, 1, 1), requires_grad=False)
        self.weight = nn.Parameter(bn_init.weight.clone().detach().view(1, -1, 1, 1), requires_grad=True)  # 可训练
        self.bias = nn.Parameter(bn_init.bias.clone().detach().view(1, -1, 1, 1), requires_grad=True)    # 可训练
        
        # 其他参数
        self.eps = 1e-5
        self.prune_q = 0.2  # 示例：剪枝20%通道
        self.calibrate_mode = False
        self.alpha = datta_alpha
        self.full_matched = True
        self.reset_mean = True
        
        # 日志配置
        os.makedirs(log_dir, exist_ok=True)
        MyBatchNorm._global_bn_counter += 1
        self.layer_id = MyBatchNorm._global_bn_counter
        self.log_path = os.path.join(log_dir, f"bn_layer_{self.layer_id:03d}.txt")
        with open(self.log_path, "w") as f:
            f.write(f"=== BN Layer {self.layer_id} Logging Start ===\n")
            f.write(f"[Init] num_features: {num_features}\n")
            f.write(f"[Init] weight shape: {self.weight.shape}\n")
    
    def reset_statistic(self):
        self.reset_mean = True
    
    def set_alpha(self, alpha):
        self.reset_mean = True
        self.alpha = alpha
    
    def get_soft_alignment_loss_weight(self):
        return torch.sum((self.weight - self.weight.detach())**2) + torch.sum((self.bias - self.bias.detach())**2)
    
    def forward_w_update_stats(self, X):
        """更新统计量（保留梯度链路）"""
        if self.reset_mean:
            self.reset_mean = False
            return
        
        # 计算batch统计量
        batch_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True)
        batch_var = torch.var(X, dim=(0, 2, 3), keepdim=True, unbiased=False)
        
        # KL散度计算自适应alpha
        dist = gauss_symm_kl_divergence(batch_mean, batch_var, self.mu, self.sigma, self.eps)
        adaptive_alpha = 1. - torch.exp(-1.0 * dist.mean())
        self.alpha = adaptive_alpha.item()
        
        # 更新mu和sigma
        if self.alpha > 0.0:
            self.full_matched = False
            self.mu.data = self.alpha * batch_mean + (1 - self.alpha) * self.mu.data
            self.sigma.data = self.alpha * batch_var + (1 - self.alpha) * self.sigma.data
        
        # 梯度修正（可选）
        gradient_mean = 2 * (self.mu - self.running_mean)
        target_std = torch.sqrt(self.sigma + self.eps)
        source_std = torch.sqrt(self.running_var + self.eps)
        gradient_std = 2 * target_std - 2 * source_std
        target_std = target_std - self.alpha * gradient_std
        self.mu.data = self.mu.data - self.alpha * gradient_mean
        self.sigma.data = target_std ** 2 - self.eps
    
    def forward(self, X):
        self.forward_w_update_stats(X)
        # 调用自定义Autograd Function（传递weight/bias作为参数）
        return StochCacheFunction.apply(self, True, X, self.weight, self.bias)

# ==================== 测试代码 ====================
def test_my_bn():
    """测试自定义BN层参数更新"""
    # 1. 创建原始BN层
    bn_original = nn.BatchNorm2d(16)  # 16个通道
    # 2. 替换为自定义BN层
    bn_custom = MyBatchNorm(bn_original)
    # 3. 创建优化器（仅优化自定义BN的weight和bias）
    optimizer = torch.optim.SGD([bn_custom.weight, bn_custom.bias], lr=0.01)
    
    # 4. 前向+反向+更新
    for i in range(5):
        optimizer.zero_grad()  # 清空梯度
        x = torch.randn(4, 16, 32, 32)  # batch=4, channels=16, H=W=32
        y = bn_custom(x)
        loss = y.mean()  # 简单的损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        # 打印参数变化（验证是否更新）
        print(f"Step {i+1}:")
        print(f"  weight mean: {bn_custom.weight.mean().item():.6f}")
        print(f"  bias mean: {bn_custom.bias.mean().item():.6f}")
        print(f"  loss: {loss.item():.6f}")

if __name__ == "__main__":
    test_my_bn()