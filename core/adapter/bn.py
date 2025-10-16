import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
import torch.nn.functional as F

def batch_norm(mean, var, X, weight, bias, eps):

    X_hat = (X - mean) / torch.sqrt(var + eps)

    Y = weight * X_hat + bias  # Scale and shift

    return Y


class MyBatchNorm(nn.Module):

    def __init__(self, bn_init: nn.BatchNorm2d, datta_alpha=0.3):
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
        self.register_buffer("mu", torch.ones(1, num_features, 1, 1))
        self.register_buffer("sigma", torch.zeros(1, num_features, 1, 1))

        self.reset_mean = True
        self.alpha = datta_alpha

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
            alpha = self.alpha
            self.mu.data = (1 - alpha) * torch.mean(X, dim=(0, 2, 3), keepdim=True).clone() + alpha * self.running_mean.data.clone()
            self.sigma.data = (1 - alpha) * torch.mean((X - self.mu) ** 2, dim=(0, 2, 3), keepdim=True).clone() + alpha * self.running_var.data.clone()

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
        super(BN, self).__init__(cfg, model, optimizer)
        return

    def forward_and_adapt(self, batch_data, model, optimizer):
        outputs = model(batch_data)
        return outputs
    def reset(self):
        for m in self.model.modules():
            if isinstance(m, MyBatchNorm):
                m.reset_statistic()
                
    def replace_bn_with_custom(self, model: nn.Module, custom_bn):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, custom_bn(module))
            else:
                self.replace_bn_with_custom(module, custom_bn)
        return model

    def configure_model(self, model: nn.Module):
        model = self.replace_bn_with_custom(model, MyBatchNorm)
        model.requires_grad_(False)
       
        return model