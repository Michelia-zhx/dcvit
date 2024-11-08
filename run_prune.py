import os
import sys
import logging

sys.path.append(".")
import prune
from src.cli import parse_dict_args
from src.run_context import RunContext


parameters = {
    # Technical details
    'gpu_id': '4',
    'omp_threads': 4,
    # 'distributed': True,
    # 'dist_url': 'tcp://127.0.0.1:29502',
    # 'rank': 0,
    # 'workers': 32,

    'seed': 2022,
    'model': 'vit_base_patch16_224',

    'dataset': 'imagenet_fewshot',
    'metric_dataset': 'imagenet_fewshot',
    'eval_dataset': 'imagenet',

    'FT': 'MiR',
    'spock': 'progressive',
    'finetune_loss': 'ce',

    'progressive': True,

    'num_sample': 50,
    'train_batch_size': 64,
    'test_batch_size': 100,

    'opt': 'adamw',
    'lr': 3e-5,
    'weight_decay': 1e-4,
    'sched': 'cosine',
    'clip_grad': 1.0,
    
    'epochs': 2000,
    
    'drop': 0.0,
    'drop_path': 0.0,

    'warmup_epochs': 20,
    'data_aug': True,
    'color_jitter': 0.3,
    'reprob': 0.0,

    'update_setting': 'before',
    'mlp_init': 'rand_sample',
    'test_time': 1,
}


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    prune.main(context)
