import os
import random
import logging
import numpy as np
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def set_seed(cfg):
    os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def set_logger(cfg):
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        # format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        format="%(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    # 打印设备信息
    # logger.info(
    #     "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    # 打印cfg中的配置内容
    # logger.info(cfg)
