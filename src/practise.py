import os
import gc
import argparse
import logging
import collections
from datetime import datetime
import time
import numpy as np
import torch

from .models import *
from . import dataset
from .finetune import AverageMeter, validate, metric, accuracy
from .compute_flops import compute_MACs_params
from .models.AdaptorWarp import AdaptorWarp
from .losses import build_loss
from .models.Swin import block2layer, layer2block


def Practise_one_block(rm_blocks, origin_model, origin_lat, train_loader, metric_loader, LOG, args):
    gc.collect()
    torch.cuda.empty_cache()

    pruned_model, _, pruned_lat = build_student(
        args.model, origin_model, rm_blocks, args.num_classes, LOG, args=args, 
        state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
    )
    lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
    LOG.info("=> latency reduction: {:.2f}%".format(lat_reduction))

    LOG.info("Metric w/o Recovering:")
    loss_init = metric(metric_loader, pruned_model, origin_model, LOG, args)
    score_init = loss_init / lat_reduction
    LOG.info(f'initial loss: {rm_blocks} -> {loss_init:.4f}/{lat_reduction:.2f}={score_init:.5f}')

    pruned_model_adaptor = AdaptorWarp(pruned_model)

    start_time = time.time()
    Practise_recover(train_loader, origin_model, pruned_model_adaptor, rm_blocks, LOG, args)
    LOG.info("Total time: {:.3f}s".format(time.time() - start_time))

    # LOG.info(f'pruned_model_adaptor: {pruned_model_adaptor}')

    LOG.info("Metric w/ Recovering:")
    recoverability = metric(metric_loader, pruned_model_adaptor, origin_model, LOG, args)
    score = recoverability / lat_reduction
    LOG.info(f'{rm_blocks} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}')
    
    pruned_model_adaptor.remove_all_preconv()
    pruned_model_adaptor.remove_all_afterconv()

    return pruned_model, (recoverability, lat_reduction, score, loss_init, score_init)

def Practise_all_blocks(rm_blocks, origin_model, origin_lat, train_loader, metric_loader, LOG, args):
    recoverabilities = dict()
    for rm_block in rm_blocks:
        _, results = Practise_one_block(rm_block, origin_model, origin_lat, train_loader, metric_loader, LOG, args)
        recoverabilities[rm_block] = results

    LOG.info('-' * 50)
    sort_list = []
    for block in recoverabilities:
        recoverability, lat_reduction, score, loss_init, score_init = recoverabilities[block]
        LOG.info(f"{block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
        sort_list.append([score, block])
    LOG.info('-' * 50)
    LOG.info('=> sorted')
    sort_list.sort()
    for score, block in sort_list:
        LOG.info(f"{block} -> {score:.4f}")
    LOG.info('-' * 50)
    LOG.info('-' * 50)
    sort_list = []
    for block in recoverabilities:
        recoverability, lat_reduction, score, loss_init, score_init = recoverabilities[block]
        LOG.info(f"{block} -> {loss_init:.4f}/{lat_reduction:.2f}={score_init:.5f}")
        sort_list.append([score_init, block])
    LOG.info('-' * 50)
    LOG.info('=> sorted')
    sort_list.sort()
    for score_init, block in sort_list:
        LOG.info(f"{block} -> {score_init:.4f}")
    LOG.info('-' * 50)

    LOG.info(f'=> scores of {args.model} (#data:{args.num_sample}, seed={args.seed})')
    LOG.info('Please use this seed to recover the model!')
    LOG.info('-' * 50)

    drop_blocks = []
    if args.rm_blocks.isdigit():
        for i in range(int(args.rm_blocks)):
            drop_blocks.append(sort_list[i][1])
    pruned_model, _, pruned_lat = build_student(
        args.model, drop_blocks, args.num_classes, LOG, args=args,
        state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
    )
    lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
    LOG.info(f'=> latency reduction: {lat_reduction:.2f}%')
    return pruned_model, drop_blocks


# insert block adaptors for mobilenet
def insert_one_block_adaptors_for_mobilenet(origin_model, prune_model, rm_block, params, args):
    origin_named_modules = dict(origin_model.named_modules())
    pruned_named_modules = dict(prune_model.model.named_modules())

    print('-' * 50)
    print('=> {}'.format(rm_block))
    has_rm_count = 0
    rm_channel = origin_named_modules[rm_block].out_channels
    key_items = rm_block.split('.')
    block_id = int(key_items[1])

    pre_block_id = block_id-has_rm_count-1
    while pre_block_id > 0:
        pruned_module = pruned_named_modules[f'features.{pre_block_id}']
        if rm_channel != pruned_module.out_channels:
            break
        last_conv_key = 'features.{}.conv.2'.format(pre_block_id)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        params.append({'params': conv.parameters()})
        pre_block_id -= 1
        # break

    after_block_id = block_id - has_rm_count
    while after_block_id < 18:
        pruned_module = pruned_named_modules[f'features.{after_block_id}']
        after_conv_key = 'features.{}.conv.0.0'.format(after_block_id)
        conv = prune_model.add_preconv_for_conv(after_conv_key)
        params.append({'params': conv.parameters()})
        if rm_channel != pruned_module.out_channels:
            break
        after_block_id += 1
        # break

    has_rm_count += 1
   

# insert block adaptors for resnet
def insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args):
    pruned_named_modules = dict(prune_model.model.named_modules())
    if 'layer1.0.conv2' in pruned_named_modules:
        last_conv_in_block = 'conv2'
    elif 'layer1.0.conv3' in pruned_named_modules:
        last_conv_in_block = 'conv3'
    else:
        raise ValueError("This is not a ResNet.")

    print('-' * 50)
    print('=> {}'.format(rm_block))
    layer, block = rm_block.split('.')
    rm_block_id = int(block)
    assert rm_block_id >= 1

    downsample = '{}.0.downsample.0'.format(layer)
    if downsample in pruned_named_modules:
        conv = prune_model.add_afterconv_for_conv(downsample)
        if conv is not None:
            params.append({'params': conv.parameters()})

    for origin_block_num in range(rm_block_id):
        last_conv_key = '{}.{}.{}'.format(layer, origin_block_num, last_conv_in_block)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        if conv is not None:
            params.append({'params': conv.parameters()})

    for origin_block_num in range(rm_block_id+1, 100):
        pruned_output_key = '{}.{}.conv1'.format(layer, origin_block_num-1)
        if pruned_output_key not in pruned_named_modules:
            break
        conv = prune_model.add_preconv_for_conv(pruned_output_key)
        if conv is not None:
            params.append({'params': conv.parameters()})

    # next stage's conv1
    next_layer_conv1 = 'layer{}.0.conv1'.format(int(layer[-1]) + 1)
    if next_layer_conv1 in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_conv1)
        if conv is not None:
            params.append({'params': conv.parameters()})

    # next stage's downsample
    next_layer_downsample = 'layer{}.0.downsample.0'.format(int(layer[-1]) + 1)
    if next_layer_downsample in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_downsample)
        if conv is not None:
            params.append({'params': conv.parameters()})

def insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks, params, args):
    rm_blocks_for_prune = []
    rm_blocks.sort()
    rm_count = [0, 0, 0, 0]
    for block in rm_blocks:
        layer, i = block.split('.')
        l_id = int(layer[-1])
        b_id = int(i)
        prune_b_id = b_id - rm_count[l_id-1]
        rm_count[l_id-1] += 1
        rm_block_prune = f'{layer}.{prune_b_id}'
        rm_blocks_for_prune.append(rm_block_prune)
    for rm_block in rm_blocks_for_prune:
        print(f'rm_block: {rm_block}')
        insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args)


# insert block adaptors for vit
def insert_one_block_adaptors_for_vit(prune_model, rm_block, params, LOG, args):
    pruned_named_modules = dict(prune_model.model.named_modules())
    # LOG.info(f'pruned_named_modules: {pruned_named_modules}')
    LOG.info('-' * 50)
    LOG.info('=> remove block {}'.format(rm_block))
    rm_block_id = int(rm_block)
    assert rm_block_id >= 0

    for pre_block_id in range(rm_block_id):
        last_linear_key = 'blocks.{}.mlp.fc2'.format(pre_block_id)
        if last_linear_key in pruned_named_modules:
            linear = prune_model.add_afteradaptor_for_linear(last_linear_key, args.adaptor, LOG=LOG)
            if linear is not None:
                params.append({'params': linear.parameters()})

    for after_block_id in range(rm_block_id, len(prune_model.model.blocks)):
        next_linear_key = 'blocks.{}.attn.qkv'.format(after_block_id)
        if next_linear_key in pruned_named_modules:
            linear = prune_model.add_preadaptor_for_linear(next_linear_key, args.adaptor, LOG=LOG)
            if linear is not None:
                params.append({'params': linear.parameters()})

def insert_all_adaptors_for_vit(origin_model, prune_model, rm_blocks, params, LOG, args):
    rm_blocks_for_prune = []
    # rm_blocks.sort()
    rm_count = 0
    for block in rm_blocks:
        b_id = int(block)
        prune_b_id = b_id - rm_count
        rm_count += 1
        rm_block_prune = str(prune_b_id)
        rm_blocks_for_prune.append(rm_block_prune)
    for rm_block in rm_blocks_for_prune:
        insert_one_block_adaptors_for_vit(prune_model, rm_block, params, LOG, args)


# insert block adaptors for swin
def insert_all_adaptors_for_swin(origin_model, prune_model, rm_blocks, params, LOG, args):
    rm_blocks_for_prune = []
    rm_blocks.sort()
    rm_count = 0
    for block in rm_blocks:
        b_id = int(block)
        prune_b_id = b_id - rm_count
        rm_count += 1
        rm_blocks_for_prune.append(prune_b_id)
    # insert_block_adaptors_for_swin(prune_model, rm_blocks_for_prune, params, LOG, args)
    pruned_named_modules = dict(prune_model.model.named_modules())
    # LOG.info(f'pruned_named_modules: {pruned_named_modules}')
    LOG.info('-' * 50)
    LOG.info('=> remove block {}'.format(rm_blocks_for_prune))
    min_rm_block_idx = int(min(rm_blocks_for_prune))
    max_rm_block_idx = int(max(rm_blocks_for_prune))
    assert min_rm_block_idx >= 0
    assert max_rm_block_idx >= 0

    for pre_block_idx in range(min_rm_block_idx):
        (pre_layer_id, pre_block_id) = block2layer(pre_block_idx, prune_model.model.depths)
        last_linear_key = 'layers.{}.blocks.{}.mlp.fc2'.format(pre_layer_id, pre_block_id)
        if last_linear_key in pruned_named_modules:
            linear = prune_model.add_afteradaptor_for_linear(last_linear_key, args.adaptor, LOG=LOG)
            if linear is not None:
                params.append({'params': linear.parameters()})

    for after_block_idx in range(max_rm_block_idx, prune_model.model.depth):
        (after_layer_id, after_block_id) = block2layer(after_block_idx, prune_model.model.depths)
        next_linear_key = 'layers.{}.blocks.{}.attn.qkv'.format(after_layer_id, after_block_id)
        if next_linear_key in pruned_named_modules:
            linear = prune_model.add_preadaptor_for_linear(next_linear_key, args.adaptor, LOG=LOG)
            if linear is not None:
                params.append({'params': linear.parameters()})


def Practise_recover(train_loader, origin_model, prune_model, rm_blocks, LOG, args):
    params = []

    if 'mobilenet' in args.model:
        assert len(rm_blocks) == 1
        insert_one_block_adaptors_for_mobilenet(origin_model, prune_model, rm_blocks[0], params, args)
    elif 'resnet' in args.model:
        insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks, params, args)
    elif 'vit' in args.model:
        insert_all_adaptors_for_vit(origin_model, prune_model, rm_blocks, params, LOG, args)
    elif 'deit' in args.model:
        insert_all_adaptors_for_vit(origin_model, prune_model, rm_blocks, params, LOG, args)
    elif 'swin' in args.model:
        insert_all_adaptors_for_swin(origin_model, prune_model, rm_blocks, params, LOG, args)

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("{} not found".format(args.opt))

    if args.sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(0.4 * args.epochs), gamma=0.1)
    

    recover_time = time.time()
    train(train_loader, optimizer, scheduler, prune_model, origin_model, LOG, args)
    LOG.info("compute recoverability {} takes {}s".format(rm_blocks, time.time() - recover_time))


def train(train_loader, optimizer, scheduler, model, origin_model, LOG, args):
    # Data loading code
    end = time.time()
    if args.finetune_loss == "mse":
        criterion = torch.nn.MSELoss(reduction='mean')
    elif args.finetune_loss == "ce":
        criterion = build_loss('KLLoss')
    else:
        raise ValueError("{} not found".format(args.finetune_loss))

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    model.cuda()
    model.eval()
    model.get_feat = 'pre_GAP'
    origin_model.get_feat = 'pre_GAP'

    torch.cuda.empty_cache()
    iter_nums = 0
    finish = False
    for iter_nums in range(1,args.epochs+1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            data = data.cuda()
            with torch.no_grad():
                t_output, t_features = origin_model(data)
            optimizer.zero_grad()
            output, s_features = model(data)
            if args.finetune_loss == "mse":
                loss = criterion(s_features, t_features)
            elif args.finetune_loss == "ce":
                loss = criterion(output, t_output)
            else:
                raise ValueError("{} not found".format(args.finetune_loss))

            losses.update(loss.data.item(), data.size(0))
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if iter_nums % 50 == 0:
            log_info = "Train: [{0}/{1}]\t".format(iter_nums, args.epochs)
            log_info += "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t".format(batch_time=batch_time)
            log_info += "Data {data_time.val:.3f} ({data_time.avg:.3f})\t".format(data_time=data_time)
            log_info += "Loss {losses.val:.4f} ({losses.avg:.4f})".format(losses=losses)
            LOG.info(log_info)
        
        scheduler.step()