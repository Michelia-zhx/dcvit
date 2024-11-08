import time
import torch
import torch.nn.functional as F
from src.utils import draw_attn_heatmap
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from tqdm import tqdm

from .utils import log_msg
from .losses import build_loss
from .models.Swin import block2layer
import numpy as np
import matplotlib.pyplot as plt

def end_to_end_finetune_dist(train_loader, metric_loader, test_loader, model, t_model, max_norm, LOG, args):
    # Data loading code
    end = time.time()

    LOG.info('freeze classifier.')
    model.module.freeze_classifier()

    if args.update_setting != '':
        if 'swin' in args.model:
            if args.update_setting == 'before': # freeze blocks after the last neighbour block
                for block_idx in range(model.depth):
                    (layer_id, block_id) = block2layer(str(block_idx), model.depths)
                    if (layer_id == model.module.last_neighbour_block[0] and block_id > model.module.last_neighbour_block[1]) or layer_id > model.module.last_neighbour_block[0]:
                        LOG.info(f'freeze block {(layer_id, block_id)}.')
                        for param in model.layers[layer_id].blocks[block_id].parameters():
                            param.requires_grad = False
        else:
            if args.update_setting == 'before': # freeze blocks after the last neighbour block
                for i in range(model.module.last_neighbour_block+1, len(model.module.blocks)):
                    LOG.info(f'freeze block {i}.')
                    for param in model.module.blocks[i].parameters():
                        param.requires_grad = False
            elif args.update_setting == 'inter': # only update blocks between the first and the last neighbour block
                for i in range(model.module.first_neighbour_block):
                    LOG.info(f'freeze block {i}.')
                    for param in model.module.blocks[i].parameters():
                        param.requires_grad = False
                for i in range(model.module.last_neighbour_block+1, len(model.module.blocks)):
                    LOG.info(f'freeze block {i}.')
                    for param in model.module.blocks[i].parameters():
                        param.requires_grad = False
            elif args.update_setting == 'ln':
                LOG.info(f'only update ln.')
                for name, param in model.named_parameters():
                    if "norm1" not in name and "norm2" not in name:
                        param.requires_grad = False
            elif args.update_setting == 'ffn_bias':
                LOG.info(f'only update ffn_bias.')
                for name, param in model.named_parameters():
                    if "mlp" not in name or "bias" not in name:
                        param.requires_grad = False
            elif args.update_setting == 'attn':
                LOG.info(f'only update attn')
                for name, param in model.named_parameters():
                    if "attn" not in name:
                        param.requires_grad = False
    
    res = filter(lambda p: p.requires_grad, model.parameters())
    length = 0
    for item in res:
        length += 1
    LOG.info(f"filter(lambda p: p.requires_grad, model.parameters())'s length: {length}")
            
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(0.4 * args.epochs), gamma=0.1)

    # switch to train mode
    model.train()
    model.module.get_feat = args.get_feat
    t_model.eval()
    t_model.module.get_feat = args.get_feat

    if args.FT == 'BP':
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.FT == 'MiR':
        criterion = torch.nn.MSELoss(reduction='mean').cuda(args.gpu)
    # elif args.FT == 'Contrastive':
    #     data = torch.rand(50, 3, 224, 224).cuda()
    #     output, features = t_model(data)
    #     args.s_dim = features.view(features.shape[0], -1).shape[1]
    #     args.t_dim = args.s_dim
    #     criterion = CRDLoss(args)
    #     mse_criterion = torch.nn.MSELoss(reduction='mean').cuda()

    iter_nums = 0
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for iter_nums in range(args.epochs):
        print(args.gpu, iter_nums)
        if args.distributed:
            train_loader.sampler.set_epoch(iter_nums)
            metric_loader.sampler.set_epoch(iter_nums)
            test_loader.sampler.set_epoch(iter_nums)

        model.module.get_feat = args.get_feat
        t_model.module.get_feat = args.get_feat
        # pbar = tqdm(range(len(train_loader)))
        for batch_idx, (data, target) in enumerate(train_loader):
            # measure data loading time
            data = data.cuda(args.gpu)
            target = target.cuda(args.gpu)
            data_time.update(time.time() - end)
            output, s_features = model(data)

            if args.FT == 'MiR':
                with torch.no_grad():
                    t_output, t_features = t_model(data)
                loss = criterion(s_features, t_features)
                # LOG.info(f"loss: {loss}")
            elif args.FT == 'BP':
                loss = criterion(output, target)
            if args.warmup_epochs > 0:
                loss = loss * min(iter_nums / args.warmup_epochs, 1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.data.item(), data.size(0))
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))
            lr = optimizer.param_groups[0]['lr']
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            msg = "Epoch:{}| Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | Data {data_time.val:.3f} ({data_time.avg:.3f}) | Loss {loss.val:.4f} ({loss.avg:.4f}) | Prec@1 {top1.val:.3f} ({top1.avg:.3f}) | Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                iter_nums+1,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5
            )
        print(args.gpu, iter_nums, 'end')

        #     pbar.set_description(log_msg(msg, "TRAIN"))
        #     pbar.update()
        # pbar.close()

        if args.draw_attn:
            draw_attn_heatmap(test_loader, model, 'epoch'+str(iter_nums)+'.eps', args.rm_blocks, args.attn_path, t_model.attn_map)
            validate_dist(test_loader, model, LOG)
            model.train()
            model.module.get_feat = args.get_feat
        else:
            if (iter_nums+1) % args.print_freq == 0:
                print(msg)
                LOG.info(msg)
            
            if (iter_nums+1) % args.eval_freq == 0:
                metric_loss = metric_dist(metric_loader, model, t_model, LOG, args)
                validate_dist(test_loader, model, LOG)
                model.train()
                model.module.get_feat = args.get_feat
        
        scheduler.step()
    
    if args.epochs % args.eval_freq != 0:
        metric_loss = metric_dist(train_loader, model, t_model, LOG, args)
        # validate_dist(test_loader, model, LOG)
    return metric_loss

def validate_dist(val_loader, model, LOG):
    start_time = time.time()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()
    # LOG.info(f'model.get_feat: {model.get_feat}')
    
    end = time.time()
    pbar = tqdm(range(len(val_loader)))
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input, target = input.cuda(), target.cuda()
            # LOG.info(f'input.shape: {input.shape}')
            # compute output
            output = model(input)
            # LOG.info(f'output.shape: {output.shape}')
            # LOG.info(f'target.shape: {target.shape}')
            # assert(0)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        msg = "Test:[{0}/{1}] | Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | Loss {loss.val:.4f} ({loss.avg:.4f}) | Prec@1 {top1.val:.3f} ({top1.avg:.3f}) | Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
            i,
            len(val_loader)-1,
            batch_time=batch_time,
            loss=losses,
            top1=top1,
            top5=top5
        )
        pbar.set_description(log_msg(msg, "TEST"))
        pbar.update()

        if i % 100 == 0 or i+1 == len(val_loader):
            LOG.info(msg)
    pbar.close()

    LOG.info('--- TEST * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} * ---'
          .format(top1=top1, top5=top5))
    LOG.info("--- testing epoch in {} seconds ---".format(time.time() - start_time))

    return top1.avg

def print_sorted(name, param_dict, LOG):
    sort_list = []
    LOG.info(f"==> {name}")
    for block in param_dict:
        avg = param_dict[block]
        # LOG.info(f"{block} -> {avg:.4f}")
        sort_list.append([avg, block])
    LOG.info('-' * 50)
    LOG.info('=> sorted')
    sort_list.sort()
    for avg, block in sort_list:
        LOG.info(f"{block} -> {avg:.4f}")
    LOG.info('-' * 50)
    LOG.info('-' * 50)

def metric_dist(metric_loader, model, origin_model, LOG, args):
    def metric_ln(model):
        attn_ln_weight = dict()
        attn_ln_bias = dict()
        ffn_ln_weight = dict()
        ffn_ln_bias = dict()
        ffn_ln_w_b = dict()
        for i in range(len(model.module.blocks)):
            if 'blocks.'+str(i)+'.norm1.weight' in model.state_dict().keys():
                attn_ln_weight[str(i)] = torch.mean(model.state_dict()['blocks.'+str(i)+'.norm1.weight']).data.item()
                attn_ln_bias[str(i)] = torch.mean(model.state_dict()['blocks.'+str(i)+'.norm1.bias']).data.item()
                ffn_ln_weight[str(i)] = torch.mean(model.state_dict()['blocks.'+str(i)+'.norm2.weight']).data.item()
                ffn_ln_bias[str(i)] = torch.mean(model.state_dict()['blocks.'+str(i)+'.norm2.bias']).data.item()
                ffn_ln_w_b[str(i)] = torch.mean(model.state_dict()['blocks.'+str(i)+'.norm2.weight']).data.item() + torch.norm(model.state_dict()['blocks.'+str(i)+'.norm2.bias']).data.item()
        # LOG.info(f"==> shape of ln weight: {model.state_dict()['blocks.0.norm1.weight'].shape}")
        # LOG.info(f"==> shape of ln bias: {model.state_dict()['blocks.0.norm1.bias'].shape}")

        print_sorted("attn_ln_weight", attn_ln_weight, LOG)
        print_sorted("attn_ln_bias", attn_ln_bias, LOG)
        print_sorted("ffn_ln_weight", ffn_ln_weight, LOG)
        print_sorted("ffn_ln_bias", ffn_ln_bias, LOG)
        print_sorted("ffn_ln_w_b", ffn_ln_w_b, LOG)
    
    # metric_ln(origin_model)
    # metric_ln(model)

    start_time = time.time()
    if args.finetune_loss == "mse":
        criterion = torch.nn.MSELoss(reduction='mean')
    elif args.finetune_loss == "ce":
        criterion = build_loss('KLLoss')
    else:
        raise ValueError("{} not found".format(args.finetune_loss))

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    origin_model.module.get_feat = args.get_feat
    model.cuda()
    model.eval()
    model.module.get_feat = args.get_feat
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # max_pt = AverageMeter()

    end = time.time()
    pbar = tqdm(range(len(metric_loader)))
    for i, tup in enumerate(metric_loader):
        if args.eval_dataset == 'cifar100':
            (data, _, _) = tup
        else:
            (data, _) = tup
        with torch.no_grad():
            data = data.cuda()
            data_time.update(time.time() - end)
            t_output, t_features = origin_model(data)
            s_output, s_features = model(data)

            if args.finetune_loss == "mse":
                loss = criterion(s_features, t_features)
            elif args.finetune_loss == "ce":
                loss = criterion(s_output, t_output)
            else:
                raise ValueError("{} not found".format(args.finetune_loss))
        
        # max_pt.update(maxp.data.item(), data.size(0))
        losses.update(loss.data.item(), data.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        msg = "Test:[{0}/{1}] | Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | Data {data_time.val:.3f} ({data_time.avg:.3f}) | Loss {loss.val:.4f} ({loss.avg:.4f})".format(
            i,
            len(metric_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses
        )
        pbar.set_description(log_msg(msg, "METRIC"))
        pbar.update()
    pbar.close()

    LOG.info(' * Metric Loss {loss.avg:.5f}'.format(loss=losses))
    LOG.info("--- testing epoch in {} seconds ---".format(time.time() - start_time))
    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
