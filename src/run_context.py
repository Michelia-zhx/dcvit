from datetime import datetime
import logging
import os
import socket

import torch
import torch.distributed as dist
import numpy as np
import random

from .cli import parse_dict_args
from .log import get_logger
from .utils import draw_attn_heatmap, get_activation, design_save_path

class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, parameters_dict, log=True, tensorboard=False):
        self.args = parse_dict_args(**parameters_dict)
        len_rm_blocks = 0

        if self.args.progressive == False:
            self.args.save_log = os.path.join(self.args.save_log, "{}_{}_{}/{}_{}/metric_{}/metric_blk".format(
                self.args.model, self.args.dataset, self.args.FT, self.args.num_sample, self.args.seed, self.args.metric_dataset))
        else:
            self.args.save_log = os.path.join(self.args.save_log, "{}_{}_{}/{}_{}/metric_{}/prune".format(
                self.args.model, self.args.dataset, self.args.FT, self.args.num_sample, self.args.seed, self.args.metric_dataset))
        self.args.save = os.path.join(self.args.save, "{}_{}_{}/{}_{}/models".format(
            self.args.model, self.args.dataset, self.args.FT, self.args.num_sample, self.args.seed))
        
        if not os.path.exists(self.args.save):
            os.makedirs(self.args.save)
        if not os.path.exists(self.args.save_log):
            os.makedirs(self.args.save_log)

        if self.args.progressive == False:
            self.logfile = os.path.join(self.args.save_log, 'loss{}_compratio{}_epoch{}_trail{}.txt'.format(self.args.finetune_loss, self.args.comp_ratio, self.args.epochs, self.args.trail))
        else:
            self.logfile = os.path.join(self.args.save_log, 'loss{}_compratio{}_rm{}_epoch{}_trail{}.txt'.format(self.args.finetune_loss, self.args.comp_ratio, self.args.best_blocks, self.args.epochs, self.args.trail))
            
        if log:
            print(f"\nrun_context: result dir = {self.args.save}\n")
            print(f"\nrun_context: log file = {self.logfile}\n")
            # print(f"\nrun_context: save path = {self.args.save_path}\n")

        # self.logger = self.init_log()
        self._init_env()
        self.vis_log = Tensorboard_logger(self.args.save + "/TB_log", not tensorboard)
        set_random_seed(self.args.seed)

        # only support 1 node
        self.args.world_size = 1

    def _init_env(self):
        os.environ['OMP_NUM_THREADS'] = str(self.args.omp_threads)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_id

    def init_log(self):
        # logging.basicConfig(level=logging.INFO, format='%(message)s')
        # logger = logging.getLogger('main')
        # FileHandler = logging.FileHandler(os.path.join(self.result_dir, f'log.txt'))
        # logger.addHandler(FileHandler)
        logger = get_logger('main', log_file=self.logfile)
        return logger

    def init_dist(self, gpu, ngpus_per_node):
        args = self.args
        args.gpu = gpu
        args.ngpus_per_node = ngpus_per_node
        print("=> use GPU: {} for training".format(args.gpu))
        if args.distributed:
            if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + gpu
            dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

    def get_rank(self):
        if not is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0

class Tensorboard_logger(object):
    """docstring for Tensorboard_logger"""
    def __init__(self, save_dir, isNone=False):
        super(Tensorboard_logger, self).__init__()
        self.save_dir = save_dir
        hostname = socket.gethostname()
        if 'GPU2' in hostname or 'gpu1' in hostname or isNone:
            self.logger = None
        else:
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter(save_dir)

    def add_scalar(self, name, value, step):
        if self.logger:
            self.logger.add_scalar(name, value, step)

    def add_text(self, name, value):
        if self.logger:
            self.logger.add_text(name, value)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True