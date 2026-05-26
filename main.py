import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing
from setproctitle import setproctitle
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

from core.configs import cfg
from core.adapter import build_adapter
from core.data import build_loader
from core.model import build_model
from core.optim import build_optimizer
from core.utils import mkdir, set_random_seed, setup_logger


def test_time_adaptation(config):
    logger = logging.getLogger("TTA.test_time")
    model = build_model(config)
    optimizer = build_optimizer(config)

    adapter_class = build_adapter(config)
    tta_model = adapter_class(config, model, optimizer)
    tta_model.cuda()

    loader = build_loader(
        config,
        config.CORRUPTION.DATASET,
        config.CORRUPTION.TYPE,
        config.CORRUPTION.SEVERITY,
    )

    domain_class_correct = defaultdict(lambda: defaultdict(int))
    domain_class_total = defaultdict(lambda: defaultdict(int))
    num_domains = len(loader.dataset.domain_id_to_name)
    progress_bar = tqdm(loader)
    running_correct = 0
    running_total = 0
    tta_model.eval()

    for batch_id, batch in enumerate(progress_bar):
        images = batch["image"]
        labels = batch["label"]
        domains = batch["domain"]

        if len(labels) == 1:
            continue

        images = images.cuda()
        labels = labels.cuda()

        logits = tta_model(images)
        predictions = torch.argmax(logits, dim=1)
        correct_mask = predictions == labels

        running_correct += correct_mask.long().sum().item()
        running_total += labels.numel()

        prediction_list = predictions.cpu().tolist()
        label_list = labels.cpu().tolist()
        domain_list = domains.cpu().tolist()

        for prediction, label, domain in zip(prediction_list, label_list, domain_list):
            domain_class_total[domain][label] += 1
            if prediction == label:
                domain_class_correct[domain][label] += 1

        if batch_id % 10 == 0:
            current_accuracy = running_correct / running_total
            if hasattr(tta_model, "mem"):
                progress_bar.set_postfix(acc=current_accuracy, bank=tta_model.mem.get_occupancy())
            else:
                progress_bar.set_postfix(acc=current_accuracy)
        torch.cuda.empty_cache()

    domain_class_average = np.zeros(num_domains)

    result_lines = []
    for domain in range(num_domains):
        class_accuracies = []

        for label in domain_class_total[domain]:
            total = domain_class_total[domain][label]
            correct = domain_class_correct[domain][label]
            class_accuracies.append(correct / total)

        if len(class_accuracies) > 0:
            domain_class_average[domain] = np.mean(class_accuracies)
            result_lines.append("%d %.2f" % (domain, domain_class_average[domain] * 100.0))
        else:
            result_lines.append("%d 0.00 (no valid classes)" % domain)

    result_lines.append("Avg: %.2f" % (domain_class_average.mean() * 100.0))
    logger.info("per domain catAvg:\n" + "\n".join(result_lines))


def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        '-ocfg',
        '--order-config-file',
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str)
    parser.add_argument(
        '-pcfg',
        '--protocol-config-file',
        metavar="FILE",
        default="",
        help="path to protocol config file",
        type=str)
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if not args.order_config_file == "":
        cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_file(args.protocol_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    dataset_name = cfg.CORRUPTION.DATASET
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA:{dataset_name:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TTA', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    logger.info(f"Loaded configuration file: \n"
                f"\tadapter: {args.adapter_config_file}\n"
                f"\tdataset: {args.dataset_config_file}\n"
                f"\torder: {args.order_config_file}")
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)

    test_time_adaptation(cfg)


if __name__ == "__main__":
    main()
