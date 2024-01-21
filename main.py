import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import sys
from config.options import *
import logging
from s2c_dataset import S2C_Dataset

from blip_reward import BlipReward
import wandb
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.backends import cudnn
from config.options import *


BEST_CHECKPOINT_NAME = "best_reward.pt"

def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True



def main(rank, config):
    # 1. 分布式训练
    # if config.distributed:
    #     torch.distributed.init_process_group(backend="nccl")
    #     local_rank = torch.distributed.get_rank()
    #     torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    #     init_seeds(config.seed + local_rank)
    # else:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     init_seeds(config.seed)

    # 2. load dataset
    train_dataset = S2C_Dataset(config.train_json_path)
    val_dataset = S2C_Dataset(config.val_json_path)
    test_dataset = S2C_Dataset(config.test_json_path)

    print(train_dataset[0])
    


if __name__ == "__main__":

    main(None, config)