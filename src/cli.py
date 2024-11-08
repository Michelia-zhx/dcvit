import re
import argparse
import logging
import sys

from . import models


LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args', 'arg2str', 'arg2strlog']


def create_parser():
    # Prune settings
    parser = argparse.ArgumentParser(description='Accelerate networks by PRACTISE')
    parser.add_argument('--dataset', type=str, default='imagenet_fewshot',
                        choices=['imagenet_fewshot', 'ADI_fewshot'],
                        help='train dataset (default: imagenet_fewshot)')
    parser.add_argument('--eval-dataset', type=str, default='imagenet', 
                        help='eval dataset (default: imagenet)')
    parser.add_argument('--metric-dataset', type=str, default='place365_sub', 
                        choices=['place365_sub', 'imagenet_fewshot', 'ADI_val', 'DI_gen'],
                        help='metric dataset (default: place365_sub)')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 32)')
    # parser.add_argument('--datafree', action='store_true',
    #                     help='whether to use other dataset to validate finetuned pruned model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--num-sample', type=int, default=50,
                        help='number of samples for training')
    parser.add_argument('--test-time', type=int, default=500, metavar='N',
                        help='eval speed time, default=500')
    
    # Distributed
    parser.add_argument('--omp-threads', default=4, type=int,
                        metavar='N', help="omp-threads")
    parser.add_argument('--gpu-id', default='7', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--distributed', default=False, type=str2bool,
                        help='distributed training', metavar='BOOL')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:FREEPORT', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--rank', type=int, default=-1)
    
    # model settings
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.05, metavar='PCT',
                        help='Drop path rate (default: 0.05)')

    # Optimizer parameters for vit
    parser.add_argument('--opt', default='sgd', type=str,
                        help='opt method (default: sgd)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"', choices=['cosine', 'linear'])
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    # parser.add_argument('--warmup_iters', type=float, default=1e-6, metavar='LR',
    #                     help='warmup iteration (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Parameters for finetuning settings
    parser.add_argument('--rm-blocks', default='', type=str,
                        help='names of removed blocks, split by comma')
    parser.add_argument('--load-rm-blocks', default='', type=str,
                        help='names of removed blocks, split by comma')
    parser.add_argument('--rm-num', default=-1, type=int,
                        help='number of blocks to remove, if -1, remove as less as possible')
    parser.add_argument('--min-block', default=1, type=int,
                        help='min block index to remove')
    parser.add_argument('--max-block', default=10, type=int,
                        help='max block index to remove')
    parser.add_argument('--best-blocks', default='', type=str,
                        help='names of best blocks to be removed, split by comma')
    # Parameters for finetuning
    parser.add_argument('--finetune-loss', default='ce', type=str,
                        help='criterien for finetuning', choices=['', 'mse', 'ce'])
    # Parameters for fintuning
    parser.add_argument('--FT', default='', type=str,
                        help='method for finetuning', choices=['', 'BP', 'MiR'])
    parser.add_argument('--spock', default='', type=str,
                        help='method for spock', choices=['', 'spock', 'total', 'once', 'progressive', 'replace'])
    parser.add_argument('--update-setting', default='', type=str,
                        help='set parts of model to be updated', choices=['','inter','before','ln','ffn_bias','attn'])
    parser.add_argument('--get-feat', default='pre_GAP', type=str,
                        help='set parts of model to be updated', choices=['', 'pre_GAP','after_GAP','middle_block'])

    # Other parameters
    parser.add_argument('--teacher', default='', type=str, metavar='PATH',
                        help='path to the pretrained teacher model (default: none)')
    parser.add_argument('--save', default='/mnt/data3/zhanghx/dcvit/results', type=str, metavar='PATH',
                        help='path to save pruned model (default: results)')
    parser.add_argument('--save_log', default='results', type=str, metavar='PATH',
                        help='path to save pruned model (default: results)')
    parser.add_argument('--state-dict-path', default='', type=str, metavar='PATH',
                        help='path to save pruned model')
    parser.add_argument('--draw-attn', action='store_true',
                        help='whether to draw attention heatmap (default: False)')
    parser.add_argument('--layertime', action='store_true',
                        help='whether to draw attention heatmap (default: False)')
    parser.add_argument('--attn-path', default='./attn_map', type=str, metavar='PATH',
                        help='path to save attention heatmap (default: attn_map)')
    parser.add_argument('--no-pretrained', default=False, type=str2bool,
                        help='do not use pretrained weight', metavar='BOOL')
    parser.add_argument('--eval-freq', '-e', default=1000, type=int,
                        metavar='N', help='evaluation frequency (default: 10)')
    parser.add_argument('--attn-freq', '-a', default=1, type=int,
                        metavar='N', help='draw attention frequency (default: 1)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--trail', default=1, type=int,
                        metavar='N', help='trail')
    # parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')

    ######## Parameters for DeiT
    # # Model ema
    # parser.add_argument('--model-ema', action='store_true')
    # parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    # parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    # parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Augmentation parameters for deit
    parser.add_argument('--data-aug', default=True, type=str2bool,
                        help='whether to do data augmentation as deit')
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. v0 or original. (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # # * Mixup params
    # parser.add_argument('--mixup', type=float, default=0.1,
    #                     help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    # parser.add_argument('--cutmix', type=float, default=1.0,
    #                     help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    # parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
    #                     help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    # parser.add_argument('--mixup-prob', type=float, default=1.0,
    #                     help='Probability of performing mixup or cutmix when either/both is enabled')
    # parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
    #                     help='Probability of switching to cutmix when both mixup and cutmix enabled')
    # parser.add_argument('--mixup-mode', type=str, default='batch',
    #                     help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--eval', action='store_true',
                        help='whether to validate loaded model without finetuning')
    parser.add_argument('--progressive', default=False, type=str2bool)
    parser.add_argument('--comp-ratio', type=float, default=1.0,
                        help='get model structure according to compression ratio')
    parser.add_argument('--mlp-ratio', type=float, default=4.0,
                        help='get model structure according to compression ratio')
    parser.add_argument('--mlp-init', type=str, default='rand_init',
                        help='how to intialise small mlp', choices=['rand_init','rand_sample','GELU_sample'])
    parser.add_argument('--stage', type=int, default=1,
                        help='when choose to remove the whole block, set the number of blocks to be removed')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    cmdline_args += sys.argv[1:]

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    if len(v) == 0:
        l = []
    else:
        l = v.split(",")
    return l

def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs

def str2teachers(v):
    model = str2list(v)
    for m in model:
        assert m in models.__all__, "{} not defined".format(m)
    return model

def arg2str(args):
    hparams_dict = vars(args)
    header = "| Key | Value |\n| :--- | :--- |\n"
    keys = hparams_dict.keys()
    #keys = sorted(keys)
    lines = ["| %s | %s |" % (key, str(hparams_dict[key])) for key in keys]
    hparams_table = header + "\n".join(lines) + "\n"
    return hparams_table

def arg2strlog(args):
    hparams_dict = vars(args)
    header = "=" * 20 + "\n"
    header += "arg key: value \n"
    keys = hparams_dict.keys()
    #keys = sorted(keys)
    lines = ["{}: {}".format(key, str(hparams_dict[key])) for key in keys]
    hparams_table = header + "\n".join(lines) + "\n"
    hparams_table += "=" * 20
    return hparams_table
