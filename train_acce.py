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
import torch.nn.functional as F
from torch.backends import cudnn
from config.learning_rates import get_learning_rate_scheduler
from config.options import *

import logging
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import pdb

logger = get_logger(__name__, log_level="INFO")
"""
命令行传入参数 batch_size = 4
那么 accelerator 就是每张卡上 batch 为 4
4 张卡那么实际 batch 就是 16
"""

def loss_func(reward):
    # target: [0] * batch
    target = torch.zeros(reward.shape[0], dtype=torch.long).to(reward.device)
    loss_list = F.cross_entropy(reward, target, reduction='none')
    loss = torch.mean(loss_list)
    
    reward_diff = reward[:, 0] - reward[:, 1]
    acc = torch.mean((reward_diff > 0).clone().detach().float())

    
    return loss, loss_list, acc



def train(args):

    set_seed(args.seed)
    logging_dir = os.path.join(args.output_dir, "logs/")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("test")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. load train/valid/test dataset
    train_dataset = S2C_Dataset("/data/liutao/mac8/fixed_data/train.pth", mode='load')
    valid_dataset = S2C_Dataset("/data/liutao/mac8/fixed_data/valid.pth", mode='load')
    test_dataset = S2C_Dataset("/data/liutao/mac8/fixed_data/test.pth", mode='load')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # 3. load model
    device = accelerator.device
    model = BlipReward(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, args)

    model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, scheduler
    )

    # 4. start Training
    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    """
    image_better: [batch, 3, 224, 224]
    """
    best_loss = 1e9
    losses = []
    acc_list = []
    for epoch in range(args.epochs):
        for step, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            reward = model(batch_data) #[batch, 2] 0列:better,  1 列：worse
            loss, loss_list, acc = loss_func(reward)

            # Gather the losses across all processes for logging (if we use distributed training).
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps


        

            break
            
        break



def validate(model, accelerator):
    model.eval()
    valid_loss = []
    valid_acc_list = []
    with torch.no_grad():
        for step, batch_data_package in enumerate():
            reward = model(batch_data_package)
            loss, loss_list, acc = loss_func(reward)
            valid_loss.append(loss_list)
            valid_acc_list.append(acc.item())
    return 1


    






if __name__ == "__main__":
    train(args)
   