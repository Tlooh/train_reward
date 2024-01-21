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

    train_dataset = S2C_Dataset()
    valid_dataset = S2C_Dataset()
    test_dataset = S2C_Dataset()

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,)

     # Set the training iterations.
    args.train_iters = args.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // args.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("steps_per_valid = ", steps_per_valid)

    model_pth =  "/home/khf/liutao/data/models/ImageReward.pt"
    print(f"load checkpoint from {model_pth}")
    state_dict = torch.load(model_pth, map_location='cpu')
    model = BlipReward(device).to(device)
    msg = model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, args)

    model = torch.nn.parallel.DistributedDataParallel(model)

    best_loss = 1e9
    optimizer.zero_grad()
    losses = []
    acc_list = []

    for epoch in range(args.epochs):
        print(f"===========epoch:{epoch}=============")
        for step, batch_data in enumerate(train_loader):
            print("handle data")




if __name__ == "__main__":
    
    train()
