import torch.optim as optim


def build_optimizer(cfg):
    def optimizer(params):
        if cfg.OPTIM.METHOD == "Adam":
            return optim.Adam(params,
                              lr=cfg.OPTIM.LR,
                              betas=(cfg.OPTIM.BETA, 0.999),
                              weight_decay=cfg.OPTIM.WD)
        else:
            raise NotImplementedError

    return optimizer
