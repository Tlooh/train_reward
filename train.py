import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import sys
from config.options import *
from s2c_dataset import S2C_Dataset

from blip_reward import BlipReward
import wandb
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.backends import cudnn
from config.learning_rates import get_learning_rate_scheduler
from config.options import *

import pdb

def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True


def train():
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_seeds(args.seed + local_rank)

    train_dataset = S2C_Dataset("/data/liutao/mac8/fixed_data/train.pth", mode='load')
    

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)


    for epoch in range(args.epochs):
        print(f"===========epoch:{epoch}=============")
        for step, batch_data in enumerate(train_loader):
            print("handle data")
            breakpoint()
            break
        break




if __name__ == "__main__":
    
    train()
