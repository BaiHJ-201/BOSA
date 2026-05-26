from .base_dataset import TTADatasetBase, DatumRaw, DatumList
from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc
from torchvision.datasets import ImageFolder
import os


class CorruptionCIFAR(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None
        if cfg.CORRUPTION.DATASET == "cifar10":
            self.load_image = load_cifar10c
        elif cfg.CORRUPTION.DATASET == "cifar100":
            self.load_image = load_cifar100c
        else:
            raise NotImplementedError(f"Unsupported CIFAR dataset: {cfg.CORRUPTION.DATASET}")

        self.domain_id_to_name = {}
        data_source = []
        for i_r in range(1):
            for i_s, severity in enumerate(self.severity):
                for i_c, corruption in enumerate(self.corruptions):
                    d_name = f"{corruption}_{severity}_{i_r}"
                    d_id = i_r * len(self.corruptions) * len(self.severity) + i_s * len(self.corruptions) + i_c
                    self.domain_id_to_name[d_id] = d_name
                    x, y = self.load_image(cfg.CORRUPTION.NUM_EX,
                                        severity,
                                        cfg.DATA_DIR,
                                        False,
                                        [corruption])
                    for i in range(len(y)):
                        data_item = DatumRaw(x[i], y[i].item(), d_id)
                        data_source.append(data_item)

        super().__init__(cfg, data_source)


class ImageNetV2(TTADatasetBase):
    def __init__(self, cfg):
        self.domain_id_to_name = {0: "imagenet-v2"}
        data_source = []

        root = os.path.join(cfg.DATA_DIR, "imagenetv2")
        dataset = ImageFolder(root)

        for img_path, _ in dataset.samples:
            folder_name = os.path.basename(os.path.dirname(img_path))
            real_label = int(folder_name)
            data_item = DatumList(img_path, real_label, 0)
            data_source.append(data_item)

        super().__init__(cfg, data_source)


class CorruptionImageNet(TTADatasetBase):
    def __init__(self, cfg, all_corruption, all_severity):
        all_corruption = [all_corruption] if not isinstance(all_corruption, list) else all_corruption
        all_severity = [all_severity] if not isinstance(all_severity, list) else all_severity

        self.corruptions = all_corruption
        self.severity = all_severity
        self.load_image = None
        if cfg.CORRUPTION.DATASET == "imagenet":
            self.load_image = load_imagenetc
        self.domain_id_to_name = {}
        data_source = []
        for i_s, severity in enumerate(self.severity):
            for i_c, corruption in enumerate(self.corruptions):
                d_name = f"{corruption}_{severity}"
                d_id = i_s * len(self.corruptions) + i_c
                self.domain_id_to_name[d_id] = d_name
                x, y = self.load_image(cfg.CORRUPTION.NUM_EX,
                                       severity,
                                       cfg.DATA_DIR,
                                       False,
                                       [corruption],
                                       prepr=lambda x: x)
                for i in range(len(y)):
                    data_item = DatumList(x[i], y[i].item(), d_id)
                    data_source.append(data_item)

        super().__init__(cfg, data_source)
