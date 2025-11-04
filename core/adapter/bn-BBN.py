import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from ..utils.bn_layers import BalancedRobustBN2dV5, BalancedRobustBN2dEMA, BalancedRobustBN1dV5
from ..utils.utils import set_named_submodule, get_named_submodule
def obtain_label(loader, x, y, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')
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
    def forward_and_adapt(self, batch_data, y, model, optimizer):
        
        # model.eval()
        # with torch.no_grad():
        #     outputs = model(batch_data)
        #     p_l = outputs.argmax(dim=1)
        self.set_bn_label(model, y)
        model.train()
        outputs = model(batch_data)
        # confidence threshold
        confidence = torch.softmax(outputs, dim=1)
        # 记录高置信度样本索引
        high_conf_indices = (confidence.max(dim=1).values > self.theta).nonzero(as_tuple=True)[0]
        # 若有高置信度样本
        if len(high_conf_indices) > 0:
            outputs_high = outputs[high_conf_indices]
            # p_l_high = p_l[high_conf_indices]
            # 高置信度样本交叉熵损失
            # high_conf_loss = nn.CrossEntropyLoss()(outputs_high, p_l_high)
            # 熵正则项
            loss = softmax_entropy(torch.softmax(outputs_high, dim=1)).mean(0)
            # loss = high_conf_loss + entropy_loss
        else:
            loss = torch.tensor(0.0, device=outputs.device)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()         
        return outputs

    @staticmethod
    def set_bn_label(model, label=None):
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, BalancedRobustBN1dV5) or isinstance(sub_module, BalancedRobustBN2dV5) or isinstance(sub_module, BalancedRobustBN2dEMA):
                sub_module.label = label
        return

    def reset(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (BalancedRobustBN2dV5, BalancedRobustBN1dV5)):
                module.reset_statistic()
                
    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
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
                                self.cfg.ADAPTER.BN.ALPHA,
                                self.cfg.ADAPTER.BN.GAMMA
                                )
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)