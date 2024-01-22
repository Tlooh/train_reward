import os
import argparse
import yaml
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# basic settings
parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file.')
parser.add_argument('--seed', default=42, type=int)


# training settings
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
parser.add_argument('--epochs', type=int, default=10, help='')
parser.add_argument('--train_iters', type=int, default=10,
                    help='total number of iterations to train over all training runs')

# device settings
parser.add_argument('--distributed', default=False, type=bool)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7')


# training options
parser.add_argument("--load_emb", dest='load_emb', action='store_true')
parser.add_argument("--load_pair_store", dest='load_pair_store', action='store_true')
parser.add_argument("--fix_base", dest='fix_base', action='store_true')


# param loose/fix settings
parser.add_argument("--fix_rate", type=float, default=0.7)


# Learning rate scheduling.
parser.add_argument('--lr', type=float, default=5e-06,
                    help='initial learning rate')
parser.add_argument('--lr-decay-iters', type=int, default=None,
                    help='number of iterations to decay LR over,'
                        ' If None defaults to `--train-iters`*`--epochs`')
parser.add_argument('--lr-decay-style', type=str, default='cosine',
                    choices=['constant', 'linear', 'cosine', 'exponential', 'inverse_square_root'],
                    help='learning rate decay function')
parser.add_argument('--lr-decay-ratio', type=float, default=0.0)
parser.add_argument('--warmup', type=float, default=0.01,
                    help='percentage of data to warmup on (.01 = 1% of all '
                        'training iters). Default 0.01')
parser.add_argument('--adam-beta1', type=float, default=0.9)
parser.add_argument('--adam-beta2', type=float, default=0.999)
parser.add_argument('--adam-eps', type=float, default=1e-8)


# save options
parser.add_argument('--clear_visualizer', dest='clear_visualizer', action='store_true')
parser.add_argument('--output_dir', default="/data/liutao/checkpoints/blipreward")
parser.add_argument('--valid_per_epoch', type=int, default=10)


# test settings
parser.add_argument('--test_ckpt', type=str, default=None, help='ckpt absolute path')

args = parser.parse_args()

config = OmegaConf.load(args.config)

