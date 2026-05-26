import math
import torch
import torch.nn as nn
from .base_adapter import BaseAdapter

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
     
        if cfg.CORRUPTION.DATASET == "cifar10":
            self.num_class = 10
        elif cfg.CORRUPTION.DATASET == "cifar100":
            self.num_class = 100
        else:
            self.num_class = 1000
        self.e_margin = math.log(self.num_class) * cfg.ADAPTER.ECOTTA.e_margin
        self.lambda_reg = cfg.ADAPTER.ECOTTA.lambda_reg
        super(EcoTTA, self).__init__(cfg, model, optimizer)

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        for name,paras in model.named_modules():
            if 'meta_part' in name:
                paras.train()
        optimizer.zero_grad()
        set_cal_mseloss(model, True)

        outputs = model(x)
        loss_reg_all = 0.
        gamma = self.lambda_reg 
        for i, encoder in enumerate(model.encoders):
            reg_loss = encoder.btsloss * gamma
            reg_loss.backward()
            loss_reg_all += reg_loss.item()

        optimizer.step()
        optimizer.zero_grad()

        set_cal_mseloss(model, False)
        outputs = model(x)
        loss_ent = entropy_minmization(outputs, e_margin=self.e_margin)
        loss_ent.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs
    
    def configure_model(self, model: nn.Module):
        model.cuda()
        for param in model.parameters():
            param.requires_grad = False
        for meta_part in model.meta_parts:
            for nm,param in meta_part.named_parameters():
                param.requires_grad = True
        if self.num_class == 10:
            checkpoint_path = 'ckpt/ecotta_netwarok/cifar10/ecotta-cifar10c.pth'
        elif self.num_class == 100:
            checkpoint_path = 'ckpt/ecotta_netwarok/cifar100/ecotta-cifar100c.pth'
        else:
            checkpoint_path = 'ckpt/ecotta_netwarok/imagenet/ecotta-imagenet.pth'
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['net'])
        model.train()
        return model
