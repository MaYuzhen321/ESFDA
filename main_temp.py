import os
import logging
from core.eval import evaluate_ori, evaluate_ori_source
from core.config import merge_from_file, cfg
import torch
from core.setada import *

from core.utils import set_seed, set_logger
# from models.model_paper import Net
from models.model2 import Net
from models.model_paper import Net_paper

logger = logging.getLogger(__name__)


def main():
    # merge_from_file(file_path)
    # set_seed(cfg)
    # set_logger(cfg)
    device = torch.device('cuda:0')
    base_model = Net(cfg.DATASET.NUM_CHANNEL, cfg.MODEL.CLASSES)
    base_model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
    # 选择优化方案
    if cfg.MODEL.ADAPTATION == 'Energy_InforNCE':
        model = setup_energy_infornce(base_model, cfg, logger).to(device)
    elif cfg.MODEL.ADAPTATION == 'SHOT':
        model = setup_shot(base_model, cfg, logger).to(device)
    elif cfg.MODEL.ADAPTATION == 'TENT':
        model = setup_tent(base_model, cfg, logger).to(device)
    elif cfg.MODEL.ADAPTATION == 'ENERGY':
        model = setup_energy(base_model, cfg, logger).to(device)
    elif cfg.MODEL.ADAPTATION == 'source':
        model = setup_source(base_model, cfg, logger).to(device)
    elif cfg.MODEL.ADAPTATION == 'NORM':
        model = setup_norm(base_model, cfg, logger).to(device)
    elif cfg.MODEL.ADAPTATION == 'PL':
        model = setup_pl(base_model, cfg, logger).to(device)
    else:
        raise 'No such adaptation method!!!!'

    if cfg.MODEL.ADAPTATION == 'source':
        evaluate_ori_source(model, cfg, logger, device)
    else:
        evaluate_ori(model, cfg, logger, device)


if __name__ == "__main__":
    file_path = './cfg/pade.yaml'
    merge_from_file(file_path)
    set_seed(cfg)
    set_logger(cfg)
    for i in range(10):
        # print(f'实验组别：{i+1}')
        logger.info("Experiment_num: {}".format(i+1))
        main()
