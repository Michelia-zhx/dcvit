from __future__ import absolute_import
import os
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T # for simplifying the transforms
import torch.distributed as dist
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
from ..thop.profile import profile

from . import resnet
from . import mobilenetv2
from . import ViT, DeiT, ViT_spock, DeiT_spock, Swin, block
import timm
from ..compute_flops import compute_MACs_params
from ..speed import eval_speed

def build_teacher(model, num_classes, LOG, args, teacher='', cuda=True):
    LOG.info("=> building teacher...")
    print("=> building teacher...")
    if args.layertime:
        origin_model = block.testViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
        origin_model = create_model(origin_model, args=args)
        latency_data = torch.randn(512, 3, 224, 224).cuda()
        # test_speed_time = 30
        if cuda:
            latency_data = latency_data.cuda()
        latency = eval_speed(origin_model, latency_data, test_time=args.test_time) * 1000
        print(f"latency: {latency}")
        assert(0)
    elif 'resnet50' in model:
        origin_model = resnet.resnet50(pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = resnet.get_blocks_to_drop(origin_model)
    elif 'resnet34' in model:
        origin_model = resnet.resnet34(pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = resnet.get_blocks_to_drop(origin_model)
    elif 'mobilenet_v2' in model:
        origin_model = mobilenetv2.mobilenet_v2(pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = mobilenetv2.get_blocks_to_drop()
    elif 'vit' in model:
        origin_model = timm.create_model(model, pretrained=True, num_classes=num_classes)
        origin_model = ViT.rm_block_from_origin(origin_model, [], LOG, args, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = ViT.get_blocks_to_drop(origin_model)
    elif 'deit' in model:
        origin_model = timm.create_model(model, pretrained=True)
        origin_model = DeiT.rm_block_from_origin(origin_model, [], LOG, args, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = DeiT.get_blocks_to_drop(origin_model)
    elif 'swin' in model:
        origin_model = timm.create_model(model, pretrained=True)
        origin_model = Swin.rm_block_from_origin(origin_model, [], LOG, args, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = Swin.get_blocks_to_drop(origin_model)
    else:
        raise ValueError(model)

    LOG.info('=> origin_model: {}'.format(all_blocks))
    origin_model = create_model(origin_model, args=args)

    input = torch.randn(1, 3, 224, 224).cuda()
    origin_MACs, origin_Params = profile(origin_model, inputs=(input, ))
    MACs_str = f'MACs={origin_MACs:.3f}G'
    Params_str = f'Params={origin_Params:.3f}M'

    latency_data = torch.randn(64, 3, 224, 224).cuda()
    if cuda:
        latency_data = latency_data.cuda()
    latency = eval_speed(origin_model, latency_data, test_time=args.test_time) * 1000
    latency_str = f'Lat={latency:.3f}ms'

    print(f'=> origin_model: {latency_str}, {MACs_str}, {Params_str}')
    LOG.info(f'=> origin_model: {latency_str}, {MACs_str}, {Params_str}')

    return origin_model, all_blocks, latency

def build_student(model, origin_model, rm_blocks, num_classes, LOG, args, state_dict_path='', teacher='', cuda=True):
    LOG.info("=> building student...")
    print("=> building student...")

    if args.eval:  # 有 load_rm_blocks 不一定需要 eval, 但有 eval 就一定要有 load_rm_blocks
        assert(args.load_rm_blocks != '')
    if args.load_rm_blocks:
        LOG.info('=> load checkpoint of previously pruned model')
        state_dict_path = os.path.join(args.save, 'checkpoint_rm{}_mlp{}_mlpratio{:.3f}_epoch{}_trail{}.tar'.format(
                args.load_rm_blocks, args.mlp_init, args.mlp_ratio, args.epochs, args.trail))
        check_point = torch.load(state_dict_path)
        state_dict = check_point['state_dict']
        LOG.info('=> load check_point and use pretrained weight from {}'.format(state_dict_path))
        rm_blocks = args.load_rm_blocks.split(',')
    else:
        LOG.info('=> not load checkpoint of previously pruned model')

    if 'resnet50' in model:
        pruned_model = resnet.resnet50_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
    elif 'resnet34' in model:
        pruned_model = resnet.resnet34_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
    elif 'mobilenet_v2' in model:
        pruned_model = mobilenetv2.mobilenet_v2_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
    elif 'vit' in model:
        pruned_model = ViT.rm_block_from_origin(origin_model, rm_blocks, LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = ViT.get_blocks_to_drop(pruned_model)
    elif 'deit' in model:
        pruned_model = DeiT.rm_block_from_origin(origin_model, rm_blocks, LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = DeiT.get_blocks_to_drop(pruned_model)
    elif 'swin' in model:
        pruned_model = Swin.rm_block_from_origin(origin_model, rm_blocks, LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = Swin.get_blocks_to_drop(pruned_model)
    else:
        raise ValueError(model)

    if args.load_rm_blocks:
        pruned_model.load_state_dict(state_dict, strict=False)
        if not args.eval:  # 意味着需要在 loaded pruned model 的基础上继续删减 block
            origin_model = pruned_model
            if 'vit' in model:
                pruned_model = ViT.rm_block_from_origin(origin_model, args.rm_blocks.split(','), LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
                summary_data = torch.zeros((1, 3, 224, 224))
                all_blocks = ViT.get_blocks_to_drop(pruned_model)
            elif 'deit' in model:
                pruned_model = DeiT.rm_block_from_origin(origin_model, args.rm_blocks.split(','), LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
                summary_data = torch.zeros((1, 3, 224, 224))
                all_blocks = DeiT.get_blocks_to_drop(pruned_model)
            elif 'swin' in model:
                pruned_model = Swin.rm_block_from_origin(origin_model, args.rm_blocks.split(','), LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
                summary_data = torch.zeros((1, 3, 224, 224))
                all_blocks = Swin.get_blocks_to_drop(pruned_model)
            else:
                raise ValueError(model)
        else:
            assert(args.rm_blocks == '')
    
    if args.load_rm_blocks:
        LOG.info('=> previously removed blocks: {}'.format(args.load_rm_blocks))  
    LOG.info('=> remove blocks: {}'.format(args.rm_blocks))
    LOG.info('=> pruned_model: {}'.format(all_blocks))
    
    pruned_model = create_model(pruned_model, args=args)

    input = torch.randn(1, 3, 224, 224).cuda()
    pruned_MACs, pruned_Params = profile(pruned_model, inputs=(input, ))
    MACs_str = f'MACs={pruned_MACs:.3f}G'
    Params_str = f'Params={pruned_Params:.3f}M'
    
    latency_data = torch.randn(64, 3, 224, 224).cuda()
    if cuda:
        latency_data = latency_data.cuda()
    latency = eval_speed(pruned_model, latency_data, test_time=args.test_time) * 1000
    latency_str = f'Lat={latency:.3f}ms'
    print(f'=> pruned_model: {latency_str}, {MACs_str}, {Params_str}')
    LOG.info(f'=> pruned_model: {latency_str}, {MACs_str}, {Params_str}')

    # print(pruned_model.state_dict().keys())
    # assert(0)
    
    return pruned_model, None, latency


def build_dc_student(model, origin_model, rm_blocks, num_classes, LOG, args, state_dict_path='', teacher='', cuda=True):
    LOG.info("=> building dense-compressed student...")

    if args.eval:
        assert(args.load_rm_blocks != '')
    if args.load_rm_blocks:
        LOG.info('=> load checkpoint of previously pruned model')
        state_dict_path = os.path.join(args.save, 'checkpoint_rm{}_mlp{}_mlpratio{:.3f}_epoch{}_trail{}.tar'.format(
                args.load_rm_blocks, args.mlp_init, args.mlp_ratio, args.epochs, args.trail))
        check_point = torch.load(state_dict_path)
        state_dict = check_point['state_dict']
        LOG.info('=> load check_point and use pretrained weight from {}'.format(state_dict_path))
        rm_blocks = args.load_rm_blocks.split(',')
    else:
        LOG.info('=> not load checkpoint of previously pruned model')

    if 'vit' in model:
        pruned_model = ViT_spock.replace_block_from_origin(origin_model, [], rm_blocks, LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = ViT_spock.get_blocks_to_replace(pruned_model)
    elif 'deit' in model:
        pruned_model = DeiT_spock.replace_block_from_origin(origin_model, [], rm_blocks, LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = DeiT.get_blocks_to_drop(pruned_model)
    else:
        raise ValueError(model)
    
    pruned_model.GELU_array = origin_model.GELU_array

    if args.load_rm_blocks:
        pruned_model.load_state_dict(state_dict, strict=False)
        if not args.eval and args.dataset != 'imagenet':  # 意味着需要在 loaded pruned model 的基础上继续删减 block
            origin_model = pruned_model
            if 'vit' in model:
                pruned_model = ViT_spock.replace_block_from_origin(origin_model, args.load_rm_blocks.split(','), args.rm_blocks.split(','), LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
                summary_data = torch.zeros((1, 3, 224, 224))
                all_blocks = ViT_spock.get_blocks_to_replace(pruned_model)
            elif 'deit' in model:
                pruned_model = DeiT_spock.replace_block_from_origin(origin_model, args.load_rm_blocks.split(','), args.rm_blocks.split(','), LOG, args, pretrained=True, num_classes=num_classes, teacher=False)
                summary_data = torch.zeros((1, 3, 224, 224))
                all_blocks = DeiT.get_blocks_to_drop(pruned_model)
            else:
                raise ValueError(model)
        else:
            assert(args.rm_blocks == '')
    
    # print(pruned_model.state_dict()['blocks.4.mlp.fc1.weight'])
    # assert(0)

    if args.load_rm_blocks:
        LOG.info('=> previously removed blocks: {}'.format(args.load_rm_blocks))  
    LOG.info('=> remove blocks: {}'.format(args.rm_blocks))
    LOG.info('=> pruned_model: {}'.format(all_blocks))

    # origin_model = create_model(origin_model, args=args)
    pruned_model = create_model(pruned_model, args=args)

    input = torch.randn(1, 3, 224, 224).cuda()
    pruned_MACs, pruned_Params = profile(pruned_model, inputs=(input, ))
    MACs_str = f'MACs={pruned_MACs:.3f}G'
    Params_str = f'Params={pruned_Params:.3f}M'

    latency_data = torch.randn(64, 3, 224, 224).cuda()
    if cuda:
        latency_data = latency_data.cuda()
    latency = eval_speed(pruned_model, latency_data, test_time=args.test_time) * 1000
    latency_str = f'Lat={latency:.3f}ms'
    print(f'=> pruned_model: {latency_str}, {MACs_str}, {Params_str}')
    LOG.info(f'=> pruned_model: {latency_str}, {MACs_str}, {Params_str}')

    return pruned_model, None, latency


def build_models(model, rm_blocks, LOG, num_classes=1000, state_dict_path='', teacher='', no_pretrained=False):
    if state_dict_path:
        LOG.info('=> load check_point from {}'.format(state_dict_path))
        check_point = torch.load(state_dict_path)
        rm_blocks = check_point['rm_blocks']
        state_dict = check_point['state_dict']

    if 'resnet50' in model:
        origin_model = resnet.resnet50(pretrained=True, num_classes=num_classes)
        pruned_model = resnet.resnet50_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = resnet.get_blocks_to_drop(origin_model)
    elif 'resnet34' in model:
        origin_model = resnet.resnet34(pretrained=True, num_classes=num_classes)
        pruned_model = resnet.resnet34_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = resnet.get_blocks_to_drop(origin_model)
    elif 'mobilenet_v2' in model:
        origin_model = mobilenetv2.mobilenet_v2(pretrained=True, num_classes=num_classes)
        pruned_model = mobilenetv2.mobilenet_v2_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = mobilenetv2.get_blocks_to_drop()
    else:
        raise ValueError(model)

    if state_dict_path and not no_pretrained:
        LOG.info('=> use pretrained weight from {}'.format(state_dict_path))
        pruned_model.load_state_dict(state_dict, strict=False)
    else:
        LOG.info('=> use pretrained weight from teacher')

    LOG.info('=> origin_model: {}'.format(all_blocks))
    LOG.info('=> remove blocks: {}'.format(rm_blocks))

    origin_MACs, origin_Params = compute_MACs_params(origin_model, summary_data)
    pruned_MACs, pruned_Params = compute_MACs_params(pruned_model, summary_data)
    reduce_MACs = (origin_MACs-pruned_MACs) / origin_MACs * 100
    reduce_Params = (origin_Params - pruned_Params) / origin_Params * 100
    MACs_str = f'MACs={pruned_MACs:.3f}G (prune {reduce_MACs:.2f}%)'
    Params_str = f'Params={pruned_Params:.3f}M (prune {reduce_Params:.2f}%)'
    LOG.info(f'=> pruned_model: {MACs_str}, {Params_str}')

    return pruned_model, origin_model, rm_blocks

def create_model(model, detach_para=False, args=None):
    if detach_para:
        for param in model.parameters():
            param.detach_()

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model