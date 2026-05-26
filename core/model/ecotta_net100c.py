# -*- coding: utf-8 -*-
#%%
import torch
from torch.nn.modules import Module
from torch.nn import functional as F
import torch.nn as nn
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

base_model = load_model("Hendrycks2020AugMix_ResNeXt", 'ckpt', 'cifar100', ThreatModel.corruptions).cuda().eval()
# with open('analysis/resnet50.txt','w') as f:
#     f.write(str(base_model))
#     f.close()
    
#%%
print(base_model)
@torch.no_grad()
def collect_block_features(model, device="cuda"):
    model.eval()
    x = torch.randn(1, 3, 32, 32).to(device)

    feats = []
    hooks = []

    def hook_fn(_, __, output):
        feats.append(output)

    # Register a hook for each original block.
    for m in model.module_list:
        hooks.append(m.register_forward_hook(hook_fn))

    model(x)

    for h in hooks:
        h.remove()

    return feats

def infer_in_out_and_stride(feats):
    infos = []

    prev = None
    for f in feats:
        if prev is None:
            in_ch = f.shape[1]
            stride = 1
        else:
            in_ch = prev.shape[1]
            stride = prev.shape[-1] // f.shape[-1]

        out_ch = f.shape[1]
        infos.append((in_ch, out_ch, stride))
        prev = f

    return infos

class simplify_resnext_augmix(Module):
    def __init__(self, model):
        super().__init__()

        # ===== input stem =====
        self.conv1 = nn.Sequential(
            model.conv_1_3x3,
            model.bn_1,
            nn.ReLU(inplace=True)  # Explicitly add a ReLU, which is safe and common.
        )

        # ===== encoder blocks =====
        # stage_1
        self.b1_l0 = model.stage_1[0]
        self.b1_l1 = model.stage_1[1]
        self.b1_l2 = model.stage_1[2]

        # stage_2
        self.b2_l0 = model.stage_2[0]
        self.b2_l1 = model.stage_2[1]
        self.b2_l2 = model.stage_2[2]

        # stage_3
        self.b3_l0 = model.stage_3[0]
        self.b3_l1 = model.stage_3[1]
        self.b3_l2 = model.stage_3[2]

        # Flatten modules in the actual forward order.
        self.module_list = [
            self.b1_l0, self.b1_l1, self.b1_l2,
            self.b2_l0, self.b2_l1, self.b2_l2,
            self.b3_l0, self.b3_l1, self.b3_l2,
        ]

        # ===== classifier =====
        self.classifier = nn.Sequential(
            model.avgpool,
            nn.Flatten(1),
            model.classifier
        )

    def forward(self, x):
        out = self.conv1(x)

        for m in self.module_list:
            out = m(out)

        out = self.classifier(out)
       
        return out

simplified_model = simplify_resnext_augmix(base_model)

"""##  Attach meta networks"""



class conv_block(Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, stride=1):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_plane)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        # return F.relu(self.conv(x))
#
class build_meta_block(Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.meta_bn = nn.BatchNorm2d(out_ch)
        self.conv_block = conv_block(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out

class one_part_of_networks(Module):
    def __init__(self, original_part, meta_part):
        super().__init__()
        self.original_part = original_part
        self.meta_part = meta_part
        self.btsloss = None
        self.cal_mseloss = False

    def forward(self, x):
        # See Algorithm 1 in the paper (page13)
        if not self.cal_mseloss:
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1)
            out3 = self.meta_part(x)
            out = out2 + out3
        else:
            x = x.detach()
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1)
            out3 = self.meta_part(x)
            out = out2 + out3
            loss = nn.L1Loss(reduction='none')
            self.btsloss = loss(out, out1.detach()).mean()
        return out

def attach_meta_networks(simplified_model, K=3):
    # Set the number of blocks of each partition (Table 13 in the paper).
    if K==3:
        num_blocks = [3,3,3]
    if K==5:
        num_blocks = [1,2,1,2,3]
    else: ValueError

    # Get necessary informations to build convolution layers of meta networks,
    # such as, the number of channels of input and output feature from the original networks.
    for l in num_blocks:
        feats = collect_block_features(simplified_model)
        infos = infer_in_out_and_stride(feats)

        in_out_depth_s = []

        prev_out_ch = None
        start = 0

        feats = collect_block_features(simplified_model)
        infos = infer_in_out_and_stride(feats)

        in_out_depth_s = []

        prev_out_ch = None
        start = 0

        for part_idx, l in enumerate(num_blocks):
            # 1. stride: product of block strides inside the partition.
            stride = 1
            for i in range(start, start + l):
                stride *= infos[i][2]

            # 2. in_ch
            if part_idx == 0:
                # First partition: output from conv1.
                in_ch = simplified_model.conv1[0].out_channels
            else:
                in_ch = prev_out_ch

            # 3. out_ch: output of the last block in the partition.
            out_ch = infos[start + l - 1][1]

            in_out_depth_s.append((in_ch, out_ch, stride))

            prev_out_ch = out_ch
            start += l

    class ecotta_networks(Module):
        def __init__(self, simplified_model, num_blocks, in_out_depth_s):
            super().__init__()
            self.conv1 = simplified_model.conv1
            encoders = []
            start_module = 0
            for l in num_blocks:
                encoder = nn.Sequential(*simplified_model.module_list[start_module: start_module+l])
                encoders.append(encoder)
                start_module = start_module+l
            self.encoders = nn.Sequential(*encoders)
            self.classifier = simplified_model.classifier

            self.meta_parts = []
            for i in range(len(num_blocks)):
                in_ch, out_ch, stride = in_out_depth_s[i]
                meta_part = build_meta_block(in_ch, out_ch, stride)
                self.encoders[i] = one_part_of_networks(self.encoders[i], meta_part)
                self.meta_parts.append(meta_part)

        def forward(self, x):
            out = self.conv1(x)
            out = self.encoders(out)
            out = self.classifier(out)
            return out

    # Return whole networks including original and meta networks
    return ecotta_networks(simplified_model, num_blocks, in_out_depth_s)

ecotta_networks = attach_meta_networks(simplified_model, K=5)



"""##  Infer the Ecotta networks"""

# Freeze original networks and make meta networks learnable.
for param in ecotta_networks.parameters():
    param.requires_grad = False
for meta_part in ecotta_networks.meta_parts:
    for param in meta_part.parameters():
        param.requires_grad = True

# ecotta_networks = ecotta_networks.cuda()
# input = torch.rand(64, 3, 32, 32).cuda()
# output = ecotta_networks(input)

def set_cal_mseloss(networks, cal_mseloss:bool):
    for encoder in networks.encoders:
        encoder.cal_mseloss = cal_mseloss



"""# Reference codes for pre-training and adaptation
1. Warm-up: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
2. TTA learning rate and optimizer: https://github.com/mr-eggplant/EATA/blob/main/main.py
3. Entropy loss Eq.(2): https://github.com/mr-eggplant/EATA/blob/main/eata.py

"""
