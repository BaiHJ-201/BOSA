from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

def build_model(cfg):
    if cfg.CORRUPTION.DATASET not in ["cifar10", "cifar100", "imagenet", "imagenetv2"]:
        raise NotImplementedError(f"Unsupported dataset: {cfg.CORRUPTION.DATASET}")

    dataset = "imagenet" if "imagenet" in cfg.CORRUPTION.DATASET else cfg.CORRUPTION.DATASET

    if cfg.ADAPTER.NAME == "ecotta":
        if dataset == "cifar10":
            from .ecotta_net10c import ecotta_networks
        elif dataset == "cifar100":
            from .ecotta_net100c import ecotta_networks
        else:
            from .ecotta_netimage import ecotta_networks
        base_model = ecotta_networks
    else:
        base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, dataset, ThreatModel.corruptions).cuda()

    return base_model
