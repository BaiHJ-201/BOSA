import torch
import torch.nn as nn
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

        self.reset_mean = True
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
            self.reset_mean = False
            self.mu.data = self.running_mean.data.clone()
            self.sigma.data = self.running_var.data.clone()
            if getattr(self, "calibrate_mode", False):
                # 用 buffer 的特征更新统计量
                buffer_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
                buffer_var = torch.mean((X - buffer_mean) ** 2, dim=(0, 2, 3), keepdim=True).clone()
                self.mu.data = (1 - self.alpha) * buffer_mean + self.alpha * self.mu.data.clone()
                self.sigma.data = (1 - self.alpha) * buffer_var + self.alpha * self.sigma.data.clone()
            # alpha = self.alpha
            # # 当前 batch 的统计量
            # batch_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True).clone()
            # batch_var = torch.mean((X - self.mu) ** 2, dim=(0, 2, 3), keepdim=True).clone()
            # self.mu.data = (1 - alpha) * batch_mean + alpha * self.running_mean.data.clone()
            # self.sigma.data = (1 - alpha) * batch_var + alpha * self.running_var.data.clone()

        Y = batch_norm(self.mu, self.sigma, X, self.weight, self.bias, eps=self.eps)

        return Y

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
        self.alpha = cfg.ADAPTER.BN.ALPHA  
        self.theta = cfg.ADAPTER.BN.THETA 
        super(BN, self).__init__(cfg, model, optimizer)
        
        return

    def forward_and_adapt(self, batch_data, model, optimizer):
        outputs = model(batch_data)
       
        # confidence threshold
        confidence = torch.softmax(outputs, dim=1)
        outputs_above_threshold = []
        for j in range(confidence.shape[0]):
            if torch.max(confidence[j]) > self.theta:
                outputs_above_threshold.append(outputs[j])

        if len(outputs_above_threshold) != 0:
            outputs_above_threshold = torch.stack(outputs_above_threshold)
            loss = softmax_entropy(outputs_above_threshold).mean(0)
        else:
            loss = torch.tensor(0.0, device=outputs.device)  

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return outputs

    def reset(self):
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()
                
    def replace_bn_with_custom(self, model: nn.Module, custom_bn):
        for name, module in model.named_children():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
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
        
    def calibrate_with_buffer(self, memory_buffer, domain_id):
        imgs = memory_buffer[domain_id]["images"].cuda()

        # 把所有 MyBatchNorm 设为 “更新模式”
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.calibrate_mode = True  # 你可以新加这个标志

        # 一次前向传播，让每个 MyBatchNorm 自己在 forward 中用 X 计算 batch mean/var
        with torch.no_grad():
            _ = self.model(imgs)

        # 校准完成后关闭 calibrate 模式
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.calibrate_mode = False

        print(f"[BN Adapter] Calibrated BN with domain {domain_id} buffer ({len(imgs)} samples).")

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)