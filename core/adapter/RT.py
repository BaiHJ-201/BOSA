import math
import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5
from ..utils.utils import set_named_submodule, get_named_submodule
from ..utils.custom_transforms import get_tta_transforms
# def set_cal_mseloss(module: nn.Module, cal_mseloss: bool):
#     """
#     遍历模型，把所有 BNWithSideBranch 的 cal_mseloss 打开/关闭
#     """
#     for m in module.modules():
#         if isinstance(m, BNWithSideBranch):
#             m.cal_mseloss = cal_mseloss

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

class BNWithSideBranch(nn.Module):
    def __init__(self, original_bn, cfg):
        super().__init__()
        self.cfg = cfg
        # 根据原始 bn 类型选择主干和侧枝
        if isinstance(original_bn, nn.BatchNorm2d):
            self.main_bn = BalancedRobustBN2dV5(
                original_bn,
                self.cfg.CORRUPTION.NUM_CLASS,
                self.cfg.ADAPTER.TRIBE.ETA,
                self.cfg.ADAPTER.TRIBE.GAMMA
            )
            self.side_bn = BalancedRobustBN2dV5(
                original_bn,
                self.cfg.CORRUPTION.NUM_CLASS,
                self.cfg.ADAPTER.TRIBE.ETA,
                self.cfg.ADAPTER.TRIBE.GAMMA
            )
        elif isinstance(original_bn, nn.BatchNorm1d):
            self.main_bn = BalancedRobustBN1dV5(
                original_bn,
                self.cfg.CORRUPTION.NUM_CLASS,
                self.cfg.ADAPTER.TRIBE.ETA,
                self.cfg.ADAPTER.TRIBE.GAMMA
            )
            self.side_bn = BalancedRobustBN1dV5(
                original_bn,
                self.cfg.CORRUPTION.NUM_CLASS,
                self.cfg.ADAPTER.TRIBE.ETA,
                self.cfg.ADAPTER.TRIBE.GAMMA
            )
        else:
            raise RuntimeError(f"Unsupported BN type: {type(original_bn)}")

        
        self.btsloss = None
        self.btsloss_per_sample = None  # 保存逐样本 loss
        self.cal_mseloss = False

    def forward(self, x):
        out1 = self.main_bn(x)
        out2 = self.side_bn(x)
        
        self.btsloss = nn.L1Loss(reduction='none')(out2, out1.detach()).mean()
        return out2
    
def replace_bn_with_sidebranch(module: nn.Module, cfg):
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(module, name, BNWithSideBranch(child, cfg))
        else:
            replace_bn_with_sidebranch(child, cfg)

class RT(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(RT, self).__init__(cfg, model, optimizer)
        if cfg.CORRUPTION.DATASET == "cifar100":
            self.e_margin = math.log(100)*cfg.ADAPTER.RT.e_margin

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, label, model, optimizer):
        # forward
        # with torch.no_grad():
        #     model.eval()
        #     p_l = model(batch_data).argmax(dim=1)
        
        model.train()
        self.set_bn_side_label(model, label)
        outputs = model(batch_data)         

        # 计算 entropy
        entropys = softmax_entropy(outputs)
        
        # 筛选可靠样本
        filter_ids = torch.where(entropys < self.e_margin)[0]
        
        # --- entropy loss ---
        if filter_ids.numel() > 0:
            loss_ent = entropys[filter_ids].mean()
        else:
            loss_ent = torch.tensor(0.0, device=outputs.device)

        # --- reg_loss  ---
        reg_loss = 0.0
        for m in model.modules():
            if isinstance(m, BNWithSideBranch) and m.btsloss is not None:
                reg_loss += m.btsloss * self.cfg.ADAPTER.RT.Lambda

        total_loss = reg_loss + loss_ent
        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return outputs
    
    def configure_model(self, model: nn.Module):
        # 替换 BN 层为带分支的结构
        replace_bn_with_sidebranch(model, self.cfg)

        # 冻结主干参数更新
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, BNWithSideBranch):
                for p in m.side_bn.parameters():
                    p.requires_grad = True

        return model
    
    @staticmethod
    def set_bn_side_label(model, label=None):
        """
        遍历模型，把 BNWithSideBranch 中的 main_bn和side_bn 设置 label
        """
        for m in model.modules():
            if isinstance(m, BNWithSideBranch):
                m.main_bn.label = label
                m.side_bn.label = label
