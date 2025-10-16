import logging
import torch
import argparse
import sys
import os
import inspect
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
sys.path.insert(0, os.path.join(current_dir, 'tinytl', 'once-for-all'))
from core.configs import cfg
from core.utils import *
from core.model import build_model
from core.data import build_loader
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle
from sklearn.metrics import confusion_matrix
import numpy as np

import time

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from tinytl.memory_cost_profiler import profile_memory_cost

def testTimeAdaptation(cfg):
    logger = logging.getLogger("TTA.test_time")
    # model, optimizer
    model = build_model(cfg)

    optimizer = build_optimizer(cfg)

    tta_adapter = build_adapter(cfg)

    tta_model = tta_adapter(cfg, model, optimizer)
    tta_model.cuda()

    loader, processor = build_loader(cfg, cfg.CORRUPTION.DATASET, cfg.CORRUPTION.TYPE, cfg.CORRUPTION.SEVERITY)

    label_record = []
    domain_record = []

    preds = []
    gts = []

    times = []
    
    domain_num = loader.dataset.domain_id_to_name.keys().__len__()
    class_num = cfg.CORRUPTION.NUM_CLASS

    tbar = tqdm(loader)
    # profile memory cost
    # require_backward = True
    # input_size = (1, 3, 32, 32)
    # memory_cost, detailed_info = profile_memory_cost(
	# 	tta_model, input_size, require_backward, activation_bits=32, trainable_param_bits=32,
	# 	frozen_param_bits=32, batch_size=64,
	# )
    # net_info = {
	# 	'memory_cost': memory_cost / 1e6,
    # 	'param_size': detailed_info['param_size'] / 1e6,
    # 	'act_size': detailed_info['act_size'] / 1e6,
    # }
    # print(f"Memory cost: {net_info['memory_cost']:.2f} MB | Param size: {net_info['param_size']:.2f} MB | Act size: {net_info['act_size']:.2f} MB")
    prev_domain = 0
    model.eval()
    first5_labels = []
    for batch_id, data_package in enumerate(tbar):
        data, label, domain = data_package["image"], data_package['label'], data_package['domain']
        if len(label) == 1:
            torch.cuda.synchronize()
            start = time.time()
            continue  # ignore the final single point
        # -------------------- 新增部分：保存batch的label --------------------
        first5_labels.append(label.cpu().numpy().tolist())
        # ------------------------------------------------------------------
        label_record.append(label)
        domain_record.append(domain)
        data, label = data.cuda(), label.cuda()
        domain_id = int(domain[0].item())
        torch.cuda.synchronize()
        start = time.time()

        output = tta_model(data)

        torch.cuda.synchronize()
        times.extend([(time.time() - start) / len(label)] * len(label))

        predict = torch.argmax(output, dim=1)
        accurate = (predict == label)
        
        preds.extend((predict.cpu() + domain * class_num).numpy().tolist())
        gts.extend((label.cpu() + domain * class_num).numpy().tolist())
        
        processor.process(accurate, domain)

        if batch_id % 10 == 0:
            if 'tta_model' in vars() and hasattr(tta_model, "mem"):
                tbar.set_postfix(acc=processor.cumulative_acc(), bank=tta_model.mem.get_occupancy())
            else:
                tbar.set_postfix(acc=processor.cumulative_acc())
        # 检查是否切换到新域
        if prev_domain is not None and domain_id != prev_domain:
            if cfg.ADAPTER.NAME == "datta" or cfg.ADAPTER.NAME == "bn":
                tta_model.reset()
                logger.info("resetting model")
            else:
                logger.warning("not resetting model")
            prev_domain = domain_id
    # -------------------- 循环结束后：写入txt文件 --------------------
    save_dir = "./logs"
    os.makedirs(save_dir, exist_ok=True)
    label_txt_path = os.path.join(save_dir, "first5_batch_labels.txt")

    with open(label_txt_path, "w") as f:
        for batch_labels in first5_labels:
            f.write(" ".join(map(str, batch_labels)) + "\n")

    print(f"✅ 前5个batch的标签已保存到: {label_txt_path}")
    # -----------------------------------------------------------------
    processor.calculate()

    logger.info(f"All Results\n{processor.info()}")

    cm = confusion_matrix(gts, preds)
    acc_per_class = (np.diag(cm) + 1e-5) / (cm.sum(axis=1) + 1e-5)

    str_ = ""
    catAvg = np.zeros(domain_num)
    for i in range(domain_num):
        catAvg[i] = acc_per_class[i*class_num:(i+1)*class_num].mean()
        str_ += "%d %.2f\n" % (i, catAvg[i] * 100.)
    str_ += "Avg: %.2f\n" % (catAvg.mean() * 100.)
    logger.info("per domain catAvg:\n" + str_)
    # print(f"Memory cost: {net_info['memory_cost']:.2f} MB | Param size: {net_info['param_size']:.2f} MB | Act size: {net_info['act_size']:.2f} MB")
    print('average adaptation time:', np.mean(times))
    pass


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

    ds = cfg.CORRUPTION.DATASET
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA:{ds:>8s}:{adapter:<10s}")

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

    testTimeAdaptation(cfg)


if __name__ == "__main__":
    main()
