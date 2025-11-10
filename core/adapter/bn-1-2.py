import torch
import torch.nn as nn
from ..utils import memory
from .base_adapter import BaseAdapter
import torch.nn.functional as F
import os
def batch_norm(mean, var, X, weight, bias, eps):

    X_hat = (X - mean) / torch.sqrt(var + eps)

    Y = weight * X_hat + bias  # Scale and shift

    return Y 
class MyBatchNorm(nn.Module):
    _global_bn_counter = 0  # 静态计数器，确保每层有唯一 ID
    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.5, log_dir="/root/WZR/TRIBE/statistic"):
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
        self.register_buffer("mu", bn_init.running_mean.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("sigma", bn_init.running_var.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        # self.register_buffer("mu", torch.ones(1, num_features, 1, 1))
        # self.register_buffer("sigma", torch.zeros(1, num_features, 1, 1))

        self.alpha = datta_alpha
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
        self.reset_mean = True
        # self.weight.data = self.weight_init.clone().detach()
        # self.bias.data = self.bias_init.clone().detach()

    def set_alpha(self, alpha):
        self.reset_mean = True
        self.alpha = alpha

    def forward(self, X):
        if self.reset_mean:
            reset_mean = False
        else:
            # 用 buffer 的特征更新统计量
            buffer_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
            buffer_var = torch.mean((X - buffer_mean) ** 2, dim=(0, 2, 3), keepdim=True).clone()

            dist = gauss_symm_kl_divergence(
                buffer_mean, buffer_var, self.mu, self.sigma, eps=self.eps)
            adaptive_alpha = 1. - torch.exp(- 1.0 * dist.mean())
            self.alpha = adaptive_alpha.item()
            self.mu.data = self.alpha * buffer_mean + (1 - self.alpha) * self.mu.data.clone()
            self.sigma.data = self.alpha * buffer_var + (1 - self.alpha) * self.sigma.data.clone()
        Y = batch_norm(self.mu, self.sigma, X, self.weight, self.bias, eps=self.eps)
        return Y
        
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
        print("1")
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
        print("2")
        # 核心：在校准阶段完成统计量更新、损失计算、参数更新
        self.calibrate_with_buffer()
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
        # for m in model.modules():  # 25, 31 ,53
        #     if isinstance(m, MyBatchNorm):
        #         m.weight.requires_grad = True
        #         m.bias.requires_grad = True
        return model
        
    def calibrate_with_buffer(self):
        imgs, ages = self.mem.get_memory()
        if len(imgs) == 0:
            return  # 无缓存数据时跳过
        
        imgs = torch.stack(imgs)
        outputs = self.model(imgs) 
        # confidence = torch.softmax(outputs, dim=1)
        # outputs_above_threshold = []
        # for j in range(confidence.shape[0]):
        #     if torch.max(confidence[j]) > self.theta:
        #         outputs_above_threshold.append(outputs[j])
        
        # if len(outputs_above_threshold) != 0:
        #     outputs_above_threshold = torch.stack(outputs_above_threshold)
        #     loss = softmax_entropy(outputs_above_threshold).mean(0)
            
        #     if has_accum_bn_grad(self.model):
        #         self.optimizer.zero_grad()  # 先清零梯度
        #         loss.backward()
        #         self.optimizer.step()

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