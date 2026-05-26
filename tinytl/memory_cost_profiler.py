import copy
import torch
import torch.nn as nn
from ofa.utils import Hswish, Hsigmoid, MyConv2d

from ofa.utils.layers import ResidualBlock
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.mobilenetv2 import InvertedResidual
from core.utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5
from core.adapter.mert import MyBatchNorm, get_bn_cache_size
__all__ = ['count_model_size', 'count_activation_size', 'profile_memory_cost']


def count_model_size(adapter):
    """处理参数共享场景，避免重复统计"""
    model_components = {}
    # 收集所有模型组件（包括aux_model、source_model等）
    all_modules = []
    # 先添加主模型
    main_model = adapter.model
    all_modules.append(("main_model", main_model))
    # 再添加其他模块（如aux_model、source_model）
    for attr_name in dir(adapter):
        attr = getattr(adapter, attr_name)
        if isinstance(attr, torch.nn.Module) and attr_name not in ['model', 'logger']:
            all_modules.append((attr_name, attr))
    
    # 记录已统计的参数地址，避免重复计算
    counted_param_ids = set()
    total_size = 0
    
    # 遍历所有模块，统计非共享参数
    for name, module in all_modules:
        for param in module.parameters():
            param_id = id(param)
            if param_id not in counted_param_ids:
                # 计算参数存储大小（假设为float32，4字节/元素）
                param_size = param.numel() * 4  # 若为其他类型需调整（如float16为2字节）
                total_size += param_size
                counted_param_ids.add(param_id)
    
    return total_size

def count_activation_size(net, input_size=(1, 3, 224, 224), activation_bits=32):
    """
    计算模型（支持参数共享的多模型，如TRIBE中的aux_model和model）的激活值内存占用
    优化点：避免重复统计共享层（非BN层）的激活值
    """
    act_byte = activation_bits / 8
   
    # --------------------------
    # 1. 提取所有子模型
    # --------------------------
    def _get_all_submodels(net):
        models = []
        for attr_name in dir(net):
            attr = getattr(net, attr_name)
            if isinstance(attr, nn.Module):
                models.append(attr)
        return models

    all_models = _get_all_submodels(net)
    if not all_models:
        raise ValueError("未找到有效的模型")

    # --------------------------
    # 2. 深拷贝模型并保留共享层关系
    # --------------------------
    copied_models = [copy.deepcopy(m) for m in all_models]

    # --------------------------
    # 3. 标记共享层
    # --------------------------
    param_layer_map = {}
    shared_layers = set()
    for model in copied_models:
        for layer in model.modules():
            if len(list(layer.children())) > 0:
                continue
            for p in layer.parameters(recurse=False):
                pid = id(p)
                if pid in param_layer_map:
                    shared_layers.add(layer)
                    shared_layers.add(param_layer_map[pid])
                else:
                    param_layer_map[pid] = layer

    # --------------------------
    # 4. 设置 device 和输入
    # --------------------------
    def _get_device(model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    device = _get_device(copied_models[0])
    input_sizes = [input_size] * len(copied_models)

    # --------------------------
    # 5. backward 激活统计钩子
    # --------------------------
    def _count_grad(m, x, y):
        # 只统计 grad_activation，tmp_activations 不累加
        if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.tensor(x[0].numel() * act_byte, device=device)
        else:
            m.grad_activations = torch.tensor(0.0, device=device)
        m.tmp_activations = torch.tensor(0.0, device=device)

    def hook_wrapper(m, x, y):
        if m in shared_layers:
            if not hasattr(m, 'counted'):
                m.counted = True
                _count_grad(m, x, y)
            else:
                m.grad_activations = torch.tensor(0.0, device=device)
                m.tmp_activations = torch.tensor(0.0, device=device)
        else:
            _count_grad(m, x, y)

    def add_hooks(model):
        for m in model.modules():
            if len(list(m.children())) > 0:
                continue
            m.register_buffer('grad_activations', torch.tensor(0.0, device=device))
            m.register_buffer('tmp_activations', torch.tensor(0.0, device=device))
            m.counted = False
            # 只给非 MyBatchNorm 的层注册 hook
            if not isinstance(m, MyBatchNorm):
                m.register_forward_hook(hook_wrapper)

    for model in copied_models:
        model.eval()
        add_hooks(model)

    # --------------------------
    # 6. 执行一次 forward（no_grad）触发 hook
    # --------------------------
    for model, in_size in zip(copied_models, input_sizes):
        x = torch.zeros(in_size, device=device)
        with torch.no_grad():
            model(x)

    # --------------------------
    # 7. 累加 grad_activation
    # --------------------------
    total_grad_activation = torch.tensor(0.0, device=device)
    for model in copied_models:
        for m in model.modules():
            if hasattr(m, 'grad_activations'):
                total_grad_activation += m.grad_activations
    return total_grad_activation.item()


def profile_memory_cost(net, input_size=(1, 3, 224, 224), 
                        activation_bits=32, batch_size=8):
	param_size = count_model_size(net)
	activation_size = count_activation_size(net, input_size, activation_bits)

	memory_cost = activation_size * batch_size 
	return memory_cost, {'param_size': param_size, 'act_size': activation_size}
