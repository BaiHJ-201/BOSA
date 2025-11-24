import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
import torch.nn.functional as F
import os
# 是属于newbn2的代码
def batch_norm(mean, var, X, weight, bias, eps):

    X_hat = (X - mean) / torch.sqrt(var + eps)

    Y = weight * X_hat + bias  # Scale and shift

    return Y

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
    _global_bn_counter = 0  # 静态计数器，确保每层有唯一 ID
    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.5, log_dir="/root/WZR/TRIBE/statistic"):
        super().__init__()
        self.register_buffer("running_mean", bn_init.running_mean.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("running_var", bn_init.running_var.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.weight = nn.Parameter(bn_init.weight.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.bias = nn.Parameter(bn_init.bias.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

        self.eps = 1e-5
        self.register_buffer("mu", bn_init.running_mean.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("sigma", bn_init.running_var.clone().detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
       

        self.reset_mean = False
        self.alpha = datta_alpha

    def reset_statistic(self):
        self.reset_mean = True

    def forward(self, X):
        if self.reset_mean:
            # self.reset_mean = False
            # 用 buffer 的特征更新统计量
            buffer_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
            buffer_var = torch.mean((X - buffer_mean) ** 2, dim=(0, 2, 3), keepdim=True).clone()
            # dist = gauss_symm_kl_divergence(
            #     buffer_mean, buffer_var, self.mu, self.sigma, eps=self.eps)
            # adaptive_alpha = 1. - torch.exp(- 1.0 * dist.mean())
            # self.alpha = adaptive_alpha.item()
            self.mu.data = self.alpha * buffer_mean + (1 - self.alpha) * self.running_mean.data.clone()
            self.sigma.data = self.alpha * buffer_var + (1 - self.alpha) * self.running_var.data.clone()
            
        Y = batch_norm(self.mu, self.sigma, X, self.weight, self.bias, eps=self.eps)

        return Y
class BN(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        self.alpha = cfg.ADAPTER.BN.ALPHA  
        self.theta = cfg.ADAPTER.BN.THETA 
        super(BN, self).__init__(cfg, model, optimizer)
        return

    def forward_and_adapt(self, batch_data):
        outputs = self.model(batch_data)
        # confidence threshold
        confidence = torch.softmax(outputs, dim=1)
        outputs_above_threshold = []
        for j in range(confidence.shape[0]):
            if torch.max(confidence[j]) > self.theta:
                outputs_above_threshold.append(outputs[j])

        if len(outputs_above_threshold) != 0:
            outputs_above_threshold = torch.stack(outputs_above_threshold)
            softmax_out = nn.Softmax(dim=1)(outputs_above_threshold)
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            loss = softmax_entropy(outputs_above_threshold).mean(0) 
            loss -=  gentropy_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return outputs
    def reset(self):
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()
                
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
        for m in model.modules():  
            if isinstance(m, MyBatchNorm):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
        return model
        
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)