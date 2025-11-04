import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5
from ..utils.utils import set_named_submodule, get_named_submodule
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# @torch.jit.script
def gauss_symm_kl_divergence(mean1, var1, mean2, var2, eps):
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

class MectaNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, affine=True, track_running_stats=True,
                 beta=0.1, use_forget_gate=True, bn_dist_scale=1.0,
                 beta_thre=0.01, name="bn"):
        super().__init__(num_features, eps=eps, affine=affine, track_running_stats=track_running_stats)
        self.beta = beta
        self.use_forget_gate = use_forget_gate
        self.bn_dist_scale = bn_dist_scale
        self.beta_thre = beta_thre
        self.full_matched = False
        self.name = name

    def forward(self, x):
        batch_var, batch_mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False)

        with torch.no_grad():
            if self.running_mean is not None and self.running_var is not None:
                dist = gauss_symm_kl_divergence(
                    batch_mean, batch_var, self.running_mean, self.running_var, eps=self.eps)
                adaptive_beta = 1. - torch.exp(- self.bn_dist_scale * dist.mean())
            else:
                adaptive_beta = self.beta

            beta = adaptive_beta.item()

            # 更新 running 均值/方差
            self.running_mean.mul_(1 - beta).add_(batch_mean * beta)
            self.running_var.mul_(1 - beta).add_(batch_var * beta)

            # # Forget gate
            # if self.use_forget_gate:
            #     if beta > self.beta_thre:
            #         self.full_matched = False
            #         if self.weight is not None:
            #             self.weight.requires_grad = True
            #             self.bias.requires_grad = True
            #     else:
            #         self.full_matched = True
            #         if self.weight is not None:
            #             self.weight.requires_grad = False
            #             self.bias.requires_grad = False

        # # 参数重整（PreG）
        # with torch.no_grad():
        #     inv_r_std = torch.sqrt(self.running_var + self.eps)
        #     weight_hat = torch.sqrt(batch_var + self.eps) / inv_r_std
        #     bias_hat = (batch_mean - self.running_mean) / inv_r_std

        # weight_hat = self.weight * weight_hat
        # bias_hat = self.weight * bias_hat + self.bias
        y = F.batch_norm(x, self.running_mean, self.running_var, self.weight,
                                 self.bias, training=False, momentum=0., eps=self.eps)
        return y
# class BN(BaseAdapter):
#     def __init__(self, cfg, model, optimizer):
#         super(BN, self).__init__(cfg, model, optimizer)
#         return

#     @torch.enable_grad()
#     def forward_and_adapt(self, batch_data, model, optimizer):
#         outputs = model(batch_data)
#         return outputs

#     def configure_model(self, model: nn.Module):

#         model.requires_grad_(False)

#         for module in model.modules():
#             if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
#                 # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
#                 # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
#                 module.track_running_stats = False
#                 module.running_mean = None
#                 module.running_var = None
        
#         return model
class BN(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        self.theta = cfg.ADAPTER.BN.THETA 
        super(BN, self).__init__(cfg, model, optimizer)
        
        return

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data):
    
        self.model.eval()
        outputs = self.model(batch_data)
        # confidence threshold
        # confidence = torch.softmax(outputs, dim=1)
        # # 记录高置信度样本索引
        # high_conf_indices = (confidence.max(dim=1).values > self.theta).nonzero(as_tuple=True)[0]
        # # 若有高置信度样本
        # if len(high_conf_indices) > 0:
        #     outputs_high = outputs[high_conf_indices]
        #     # p_l_high = p_l[high_conf_indices]
        #     # 高置信度样本交叉熵损失
        #     # high_conf_loss = nn.CrossEntropyLoss()(outputs_high, p_l_high)
        #     # 熵正则项
        #     loss = softmax_entropy(torch.softmax(outputs_high, dim=1)).mean(0)
        #     # loss = high_conf_loss + entropy_loss
        # else:
        #     loss = torch.tensor(0.0, device=outputs.device)

        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()         
        return outputs
    def configure_model(self, model: nn.Module):
        """
        递归地将模型中的 BatchNorm2d 层替换为 MectaNorm2d。
        移除了内部函数定义，逻辑直接写在 configure_model 中。
        """
        # 冻结模型所有参数
        model.requires_grad_(False)
        # 要从原 BN 拷贝的属性
        copy_keys = ['eps', 'momentum', 'affine', 'track_running_stats']
        # 用于统计替换数量
        n_replaced = 0
        # 用一个栈或队列来手动实现递归遍历
        stack = [(model, "")]  # (当前模块, 模块路径前缀)
        while stack:
            module, name_prefix = stack.pop()
            for sub_name, sub_module in module.named_children():
                full_name = f"{name_prefix}.{sub_name}" if name_prefix else sub_name
                if isinstance(sub_module, nn.BatchNorm2d):
                    print(f"🔁 Replace BatchNorm2d -> MectaNorm2d at: {full_name}")
                    n_replaced += 1

                    # 创建新的 MectaNorm2d 层
                    new_bn = MectaNorm2d(sub_module.num_features)
                    # 拷贝原 BN 参数
                    new_bn.load_state_dict(sub_module.state_dict(), strict=False)
                    # 启用新层梯度
                    # new_bn.requires_grad_(True)
                    # 替换模块
                    setattr(module, sub_name, new_bn)
                else:
                    # 如果子模块本身包含子结构，则入栈递归遍历
                    stack.append((sub_module, full_name))
        print(f"✅ Total BatchNorm2d layers replaced with MectaNorm2d: {n_replaced}")
        return model