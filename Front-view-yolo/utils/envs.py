import sys
import os
import copy
from datetime import datetime
import logging
import torch
import random
import numpy as np


# individual packages
from .fileproc import safeMakeDirs
from .cfg_parser import getConfig


def setLogging(log_dir, log_name, stdout_flag):
    log_fp = os.path.join(log_dir, log_name)

    if stdout_flag:
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(filename=log_fp, format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


def combineConfig(cur_cfg, train_flag):
    ret_cfg = {}
    for k, v in cur_cfg.items():
        if k == 'train' or k == 'val' or k == 'test' or k == 'speed':
            continue
        ret_cfg[k] = v
    if train_flag == 0:
        key = 'train'
    elif train_flag == 1:
        key = 'val'
    elif train_flag == 2:
        key = 'test'
    else:
        key = 'speed'
    for k, v in cur_cfg[key].items():
        ret_cfg[k] = v
    return ret_cfg


def initEnv(train_flag, model_name):
    cfgs_root = 'cfgs'
    cur_cfg = getConfig(cfgs_root, model_name)

    root_dir = cur_cfg['output_root']
    cur_cfg['model_name'] = model_name
    work_dir = os.path.join(root_dir, model_name)

    if train_flag == 0:
        safeMakeDirs(work_dir)
        stdout_flag = cur_cfg['train']['stdout']
        dt = datetime.now()
        dt_str = dt.strftime('%Y-%m-%d_time_%H_%M_%S')
        sub_work_dir = os.path.join(work_dir, dt_str)
        safeMakeDirs(sub_work_dir)
        log_name = 'train_' + dt_str + '.log'
        setLogging(sub_work_dir, log_name, stdout_flag)

        gpus = cur_cfg['train']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        weight_save_dir = os.path.join(sub_work_dir, 'weights')
        safeMakeDirs(weight_save_dir)
        cur_cfg['train']['backup_dir'] = weight_save_dir

        os.system('cp -r ./brambox/ ./cfgs ./examples ./utils ./vedanet %s' % (sub_work_dir))
    elif train_flag == 1:
        stdout_flag = cur_cfg['val']['stdout']
        sub_work_dir = os.path.join(root_dir, 'val')
        safeMakeDirs(sub_work_dir)
        log_name = 'val_' + '.log'
        setLogging(sub_work_dir, log_name, stdout_flag)

        gpus = cur_cfg['val']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    elif train_flag == 2:
        stdout_flag = cur_cfg['test']['stdout']
        sub_work_dir = os.path.join(root_dir, 'test')
        safeMakeDirs(sub_work_dir)
        log_name = 'test_' + '.log'
        setLogging(sub_work_dir, log_name, stdout_flag)

        gpus = cur_cfg['test']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        gpus = cur_cfg['speed']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    ret_cfg = combineConfig(cur_cfg, train_flag)

    return ret_cfg


def randomSeeding(seed):
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)


if __name__ == '__main__':
    pass
