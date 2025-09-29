import math
import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5
from ..utils.utils import set_named_submodule, get_named_submodule
from ..utils.custom_transforms import get_tta_transforms
from ..model.wideresnet40 import attach_meta_networks
def entropy_minmization(outputs,e_margin):
    """Calculate entropy of the output of a batch of images.
    """
    # convert to probabilities
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    # ids1 = filter_ids_1
    # ids2 = torch.where(ids1[0] > -0.1)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    return loss

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def set_cal_mseloss(networks, cal_mseloss:bool):
    for encoder in networks.encoders:
        encoder.cal_mseloss = cal_mseloss

class EcoTTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(EcoTTA, self).__init__(cfg, model, optimizer)
        if cfg.CORRUPTION.DATASET == "cifar100":
            self.e_margin = math.log(100)*cfg.ADAPTER.RT.e_margin
    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, label, model, optimizer):
        optimizer_TTA.zero_grad()
        set_cal_mseloss(net, True)

        self.set_bn_side_label(model, label)
        outputs = model(batch_data)         
        loss_reg_all = 0.

        for i, encoder in enumerate(model.encoders):
            reg_loss = encoder.btsloss * self.cfg.ADAPTER.RT.Lambda
            reg_loss.backward()
            loss_reg_all += reg_loss.item()

        optimizer.step()
        optimizer.zero_grad()

        set_cal_mseloss(net, False)
        model.eval()
        outputs = model(batch_data)
        loss_ent = entropy_minmization(outputs,e_margin=self.e_margin)
        loss_ent.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return ema_out
    
    @staticmethod
    def set_bn_label(model, label=None):
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, BalancedRobustBN1dV5) or isinstance(sub_module, BalancedRobustBN2dV5) or isinstance(sub_module, BalancedRobustBN2dEMA):
                sub_module.label = label
        return

    
    def configure_model(self, model: nn.Module):
        model = attach_meta_networks(model, K=5)

        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm2d) or isinstance(sub_module, nn.BatchNorm1d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = BalancedRobustBN2dV5
                # NewBN = BalancedRobustBN2dEMA
            elif isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = BalancedRobustBN1dV5
            else:
                raise RuntimeError()
            
            momentum_bn = NewBN(bn_layer,
                                self.cfg.CORRUPTION.NUM_CLASS,
                                self.cfg.ADAPTER.TRIBE.ETA,
                                self.cfg.ADAPTER.TRIBE.GAMMA
                                )

            set_named_submodule(model, name, momentum_bn)
        # 冻结参数，只训练 meta networks
        for param in model.parameters():
            param.requires_grad = False
        for meta_part in model.meta_parts:
            for param in meta_part.parameters():
                param.requires_grad = True
        return model