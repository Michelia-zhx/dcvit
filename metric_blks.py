import os
import argparse
import logging
import collections
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import time
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# print(os.getcwd())
from src import cli
from src.run_context import RunContext
from src.models import *
from src import dataset
from src.finetune import end_to_end_finetune
from src.utils import draw_attn_heatmap, get_activation, design_save_path
from src.thop.get_structure import flops2structure
import matplotlib.pyplot as plt

def main(context):
    args = context.args

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, context))
    else:
        gpu = args.gpu_id.split(',')
        if len(gpu) == 1:
            # single gpu
            # gpu = int(gpu[0])
            gpu = 0
        else:
            # multi-gpu, use DataParallel to train
            gpu = None
        main_worker(gpu, ngpus_per_node, context)

def split_list_by_n(list_collection, n):
    for i in range(0, len(list_collection), n):
        yield list_collection[i: i + n]

def main_worker(gpu, ngpus_per_node, context):
    global args
    args = context.args
    args.gpu = gpu
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    context.init_dist(gpu, ngpus_per_node)
    
    LOG = context.init_log()
    context.vis_log.add_text('hparams', cli.arg2str(args))
    LOG.info("{}".format(cli.arg2strlog(args)))
    
    assert(args.comp_ratio < 1.0)

    if args.eval_dataset == 'cifar10':
        args.num_classes = 10
    elif args.eval_dataset == 'cifar100':
        args.num_classes = 100
    elif args.eval_dataset == 'imagenet':
        args.num_classes = 1000
    else:
        args.num_classes = 1000

    start_time = time.time()
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    if args.distributed:
        args.train_batch_size = int(args.train_batch_size / args.ngpus_per_node)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    if args.dataset == 'imagenet':
        train_loader = dataset.__dict__['imagenet'](args, train=True, batch_size=args.train_batch_size)
        args.print_freq = 1
        args.eval_freq = 1
    else: # imagenet_fewshot
        args.eval_freq = 500
        # args.attn_freq = 50
        assert args.seed > 0, "Please set seed"
        train_loader = dataset.__dict__[args.dataset](args, args.num_sample, batch_size=args.train_batch_size, seed=args.seed)
        try:
            train_loader.dataset.samples_to_file(os.path.join(args.save, "samples.txt"))
        except:
            print('Not save samples.txt')
    
    if args.metric_dataset == 'imagenet_fewshot':
        metric_loader = dataset.__dict__[args.metric_dataset](args, args.num_sample, seed=args.seed, train=False)
    else: # DI_gen, place365_sub, CUB, ADI_val
        metric_loader = dataset.__dict__[args.metric_dataset](args, num_sample=args.num_sample, batch_size=args.test_batch_size, seed=args.seed)
    
    assert(args.eval_dataset == 'imagenet')
    test_loader = dataset.__dict__[args.eval_dataset](args, train=False, batch_size=args.test_batch_size)

    LOG.info("=> load dataset in {} seconds".format(time.time() - start_time))

    LOG.info("=> original model: {}".format(args.model))
    origin_model, _, _ = build_teacher(
        args.model, args.num_classes, LOG, args, teacher=args.teacher, cuda=args.cuda
    )

    rm_num, mlp_ratio = flops2structure(origin_model, args.comp_ratio, LOG, args, args.rm_num)
    LOG.info(f"To retain {args.comp_ratio*100}% FLOPs of orginal model, replace {rm_num} blocks with mlp ratio {mlp_ratio}.")
    args.mlp_ratio = mlp_ratio

    remained_blocks = origin_model.get_blocks_to_drop()[args.min_block : args.max_block+1]
    removed_blocks = []
    for stage in range(1, 2):
        LOG.info('-' * 50)
        LOG.info(f"==> Stage: {stage}")
        LOG.info(f"==> Removed blocks: {removed_blocks}")
        LOG.info(f"==> Remained blocks: {remained_blocks}")
        LOG.info('-' * 50)
        args.load_rm_blocks = ','.join(removed_blocks)
        block2loss = dict()
        block2top1 = dict()
        block2top5 = dict()
        for block in remained_blocks:
            args.rm_blocks = block
            rm_blocks = block.split(',')
            if args.distributed:
                pruned_model, _, _ = build_dc_student(args.model, origin_model.module, rm_blocks, args.num_classes, LOG, args=args, 
                    state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
                )
            else:
                pruned_model, _, _ = build_dc_student(args.model, origin_model, rm_blocks, args.num_classes, LOG, args=args, 
                    state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
                )

            if args.distributed:
                origin_model.module.last_neighbour_block = pruned_model.module.last_neighbour_block
                LOG.info(f'last_neighbour_block: {origin_model.module.last_neighbour_block}')
            else:
                origin_model.last_neighbour_block = pruned_model.last_neighbour_block
                LOG.info(f'last_neighbour_block: {origin_model.last_neighbour_block}')
            
            LOG.info(f"LOAD: {args.load_rm_blocks}, RM: {args.rm_blocks} => start finetuning:")
            loss, top1, top5 = end_to_end_finetune(train_loader, metric_loader, test_loader, pruned_model, 
                                                       origin_model, args.clip_grad, LOG, args = args)
            
            block2loss[block] = loss
            block2top1[block] = top1
            block2top5[block] = top5

            rm_blocks, save_path = design_save_path(args.load_rm_blocks, block, args)
            if context.is_main_process():
                check_point = {
                    'state_dict': pruned_model.module.state_dict() if args.distributed else pruned_model.state_dict(),
                    'rm_blocks': rm_blocks,
                }
                torch.save(check_point, save_path)
        
        LOG.info('-' * 50)
        sort_list = []
        for block in remained_blocks:
            ce_loss = block2loss[block]
            top1 = block2top1[block]
            top5 = block2top5[block]
            LOG.info(f"{args.load_rm_blocks}, {block} -> {ce_loss:.4f}, {top1:.5f}, {top5:.5f}")
            sort_list.append([ce_loss, block, top1, top5])
        LOG.info('-' * 50)
        LOG.info('=> sorted')
        sort_list.sort()
        for ce_loss, block, top1, top5 in sort_list:
            LOG.info(f"{args.load_rm_blocks}, {block} -> {ce_loss:.4f}, {top1:.5f}, {top5:.5f}")
        LOG.info('-' * 50)

        block_minloss = min(block2loss, key = block2loss.get)
        LOG.info(f"==> Block {block_minloss} has min training ce-loss")
        remained_blocks.remove(block_minloss)
        removed_blocks.append(block_minloss)
        removed_blocks.sort()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
