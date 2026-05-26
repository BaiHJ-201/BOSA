import math

import torch
import torch.nn as nn

from .base_adapter import BaseAdapter
from ..utils.autofreeze.batch_norm import AutoFreezeNorm2d
from ..utils.autofreeze.conv import AutoFreezeConv2d
from ..utils.autofreeze.fc import AutoFreezeFC
from ..utils.custom_transforms import get_tta_transforms


AUTOFREEZE_LAYERS = (AutoFreezeNorm2d, AutoFreezeConv2d, AutoFreezeFC)


class DAS(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super().__init__(cfg, model, optimizer)
        self.transforms = get_tta_transforms(cfg)
        self.high_margin = math.log(cfg.CORRUPTION.NUM_CLASS) * cfg.ADAPTER.DAS.e_margin

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        imgs = x.unsqueeze(0) if x.dim() == 3 else x
        assert imgs.dim() == 4, f"Expect 4D input, got {imgs.shape}"

        self.set_sparsity(self.model, active=False)
        sample_idx = torch.randperm(imgs.size(0), device=imgs.device)[: min(10, imgs.size(0))]

        logits_subset = self.model(imgs[sample_idx])
        entropy(logits_subset).backward()

        importance = self.layer_importance(self.model)
        optimizer.zero_grad(set_to_none=True)
        self.set_sparsity(self.model, active=True, layer_weights=normalize_importance(importance))

        logits = self.model(imgs)
        logits_aug = self.model(self.transforms(imgs))
        self.optimizer.zero_grad(set_to_none=True)

        entropies = sample_entropy(logits)
        selected = entropies[entropies < self.high_margin]
        if selected.numel() == 0:
            entropy_loss = entropies.mean()
        else:
            weights = torch.exp(-(selected.detach() - self.high_margin))
            entropy_loss = selected.mul(weights).mean()

        loss = entropy_loss + 0.01 * logits.shape[1] * consistency(logits, logits_aug)
        loss.backward()
        self.optimizer.step()
        return logits

    def layer_importance(self, model):
        activation_sizes, total_activation_size = self.activation_sizes(model)
        importance = {}

        for name, param in model.named_parameters():
            if param.grad is None or name not in activation_sizes:
                continue

            grad_norm = torch.norm(param.grad.detach()).item()
            layer_size = activation_sizes[name]
            memory_weight = 1.0
            if total_activation_size > 0 and layer_size > 0:
                memory_weight = math.log(total_activation_size / layer_size)

            importance[name] = grad_norm / (param.grad.numel() ** 0.5) * memory_weight

        return importance

    def activation_sizes(self, model):
        sizes = {}
        total_size = 0

        for name, module in model.named_modules():
            if isinstance(module, AUTOFREEZE_LAYERS):
                param_name = f"{name}.weight"
                sizes[param_name] = module.activation_size
                total_size += module.activation_size

        return sizes, total_size

    def set_sparsity(self, model, active, layer_weights=None):
        layer_weights = layer_weights or {}

        for name, module in model.named_modules():
            if not isinstance(module, AUTOFREEZE_LAYERS):
                continue

            module.sparsity_signal = active
            if active:
                module.clip_ratio = 1.0 - layer_weights.get(f"{name}.weight", 0.0)
            else:
                module.clip_ratio = 0.0

    def collect_params(self, model):
        params = []
        names = []

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ["weight", "bias"] and param.requires_grad:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")

        return params, names

    def configure_model(self, model):
        model = self.insert_autofreeze_layers(model)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.train()
        model.requires_grad_(True)
        return model

    def insert_autofreeze_layers(self, model):
        num_bn = self.count_modules(model, nn.BatchNorm2d)
        num_conv = self.count_modules(model, nn.Conv2d)
        num_fc = self.count_modules(model, nn.Linear)

        replaced_bn = self.replace_batch_norm(model, self.cfg.MODEL.ARCH)
        replaced_conv = self.replace_conv(model, self.cfg.MODEL.ARCH)
        replaced_fc = self.replace_linear(model, self.cfg.MODEL.ARCH)

        assert replaced_bn == num_bn == self.count_modules(model, AutoFreezeNorm2d)
        assert replaced_conv == num_conv == self.count_modules(model, AutoFreezeConv2d)
        assert replaced_fc == num_fc == self.count_modules(model, AutoFreezeFC)
        return model

    def replace_batch_norm(self, model, prefix, replaced=0):
        copy_keys = ["eps", "momentum", "affine", "track_running_stats"]

        for name, module in model.named_children():
            full_name = f"{prefix}.{name}"
            if isinstance(module, nn.BatchNorm2d):
                replaced += 1
                new_module = AutoFreezeNorm2d(
                    module.num_features,
                    **{key: getattr(module, key) for key in copy_keys},
                    name=full_name,
                    num=replaced,
                    beta_thre=0,
                    BN_only=self.cfg.BN_ONLY,
                )
                new_module.load_state_dict(module.state_dict())
                setattr(model, name, new_module)
            else:
                replaced = self.replace_batch_norm(module, full_name, replaced)

        return replaced

    def replace_conv(self, model, prefix, replaced=0):
        copy_keys = ["stride", "padding", "dilation", "groups", "bias", "padding_mode"]

        for name, module in model.named_children():
            full_name = f"{prefix}.{name}"
            if isinstance(module, nn.Conv2d):
                replaced += 1
                new_module = AutoFreezeConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    **{key: getattr(module, key) for key in copy_keys},
                    name=full_name,
                    num=replaced,
                    BN_only=self.cfg.BN_ONLY,
                )
                new_module.load_state_dict(module.state_dict())
                setattr(model, name, new_module)
            else:
                replaced = self.replace_conv(module, full_name, replaced)

        return replaced

    def replace_linear(self, model, prefix, replaced=0):
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}"
            if isinstance(module, nn.Linear):
                replaced += 1
                new_module = AutoFreezeFC(
                    module.in_features,
                    module.out_features,
                    module.bias,
                    name=full_name,
                    num=replaced,
                    BN_only=self.cfg.BN_ONLY,
                )
                new_module.load_state_dict(module.state_dict())
                setattr(model, name, new_module)
            else:
                replaced = self.replace_linear(module, full_name, replaced)

        return replaced

    @staticmethod
    def count_modules(model, module_type):
        return sum(1 for module in model.modules() if isinstance(module, module_type))


def entropy(logits):
    return sample_entropy(logits).mean()


def sample_entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def consistency(logits, logits_aug):
    return -(logits.softmax(1) * logits_aug.log_softmax(1)).sum(1).mean()


def normalize_importance(importance):
    if not importance:
        return {}

    max_value = max(importance.values())
    if max_value <= 0:
        return {name: 0.0 for name in importance}

    return {name: value / max_value for name, value in importance.items()}
