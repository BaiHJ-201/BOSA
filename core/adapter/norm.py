import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
import math
from copy import deepcopy

class BayesianBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """
    @staticmethod
    def find_bns(parent, prior, cfg):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = BayesianBatchNorm(child, prior, cfg)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BayesianBatchNorm.find_bns(child, prior, cfg))
        return replace_mods

    @staticmethod
    def adapt_model(cfg, model, prior):
        replace_mods = BayesianBatchNorm.find_bns(model, prior, cfg)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior, cfg):
        super(BayesianBatchNorm, self).__init__()
        assert 0 <= prior <= 1, "prior must be in [0, 1]"
        self.layer = layer
        self.cfg = cfg
        self.layer.eval()
        # 使用 cfg.ADAPTER.NORM.MOMENTUM
        self.norm = nn.BatchNorm2d(
            self.layer.num_features,
            affine=False,
            momentum=self.cfg.ADAPTER.NORM.MOMENTUM,
            track_running_stats=True
        ).cuda()
        self.normed_div_mean = torch.zeros(1).cuda()

    def forward(self, input):
        self.norm(input)
        self.norm.eval()
        source_distribution = torch.distributions.MultivariateNormal(
            self.layer.running_mean,
            (self.layer.running_var + 1e-5) * torch.eye(self.layer.running_var.shape[0]).cuda()
        )
        target_distribution = torch.distributions.MultivariateNormal(
            self.norm.running_mean,
            (self.norm.running_var + 1e-5) * torch.eye(self.norm.running_var.shape[0]).cuda()
        )

        self.div = 0.5 * (
            torch.distributions.kl_divergence(source_distribution, target_distribution) +
            torch.distributions.kl_divergence(target_distribution, source_distribution)
        )

        self.div_values = self.div
        self.prior = self.normed_div_mean

        running_mean = (self.prior * self.layer.running_mean +
                        (1 - self.prior) * self.norm.running_mean)
        running_var = (self.prior * self.layer.running_var +
                       (1 - self.prior) * self.norm.running_var +
                       self.prior * (1 - self.prior) *
                       ((self.layer.running_mean - self.norm.running_mean) ** 2))

        output = (input - running_mean[None, :, None, None]) / torch.sqrt(
            running_var[None, :, None, None] + self.layer.eps
        ) * self.layer.weight[None, :, None, None] + self.layer.bias[None, :, None, None]

        return output
class Norm(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(Norm, self).__init__(cfg, model, optimizer)
        self.EMA_normed_div_mean = torch.zeros(int(sum(1 for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))/2)).cuda()
        self.index = 0
        return

    @torch.no_grad()
    def forward_and_adapt(self, x):
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, BayesianBatchNorm):
                    m.norm.train()
            imgs_test = x
            self.m = 1/2

            div_mean = []
            _ = self.model(imgs_test)

            for name, module in self.model.named_modules():
                if isinstance(module, BayesianBatchNorm):
                    div_mean.append(module.div_values)

            normed_div_mean = scale_to_mean_std(div_mean)
            tructed_div = [max(min(x, torch.tensor(1).cuda()), torch.tensor(-1).cuda()) for x in normed_div_mean]

            ii = 0
            for name, module in self.model.named_modules():
                if isinstance(module, BayesianBatchNorm):
                    module.normed_div_mean = (tructed_div[ii]+1)/2 * self.m
                    ii += 1
            prediction = self.model(imgs_test)

            # self.EMA_normed_div_mean = self.EMA_normed_div_mean * 0.9 + ( (torch.stack(tructed_div) + 1) / 2 * self.m) * 0.1
            # jj = 0
            # for name, module in self.model.named_modules():
            #     if isinstance(module, BayesianBatchNorm):
            #         module.normed_div_mean = self.EMA_normed_div_mean[jj]
            #         jj += 1

        return prediction

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        self.model_state = deepcopy(self.model.state_dict())
        return self.model_state, None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
                
    def configure_model(self, model: nn.Module):
        model.eval()
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.0
        model = BayesianBatchNorm.adapt_model(self.cfg, model, prior=1).cuda()
        return model
def scale_to_mean_std(numbers):
    mean = sum(numbers) / len(numbers)
    std = math.sqrt(sum((x - mean) ** 2 for x in numbers) / len(numbers))
    scaled_numbers = [(x - mean) / std for x in numbers]
    return scaled_numbers