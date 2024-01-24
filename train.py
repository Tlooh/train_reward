from config.options import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.BatchSize = args.batch_size * args.gradient_accumulation_steps * args.gpu_num
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import sys

from s2c_dataset import S2C_Dataset

from blip_reward import BlipReward
import wandb
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.backends import cudnn
from config.learning_rates import get_learning_rate_scheduler
from utils import *

import pdb

def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True



def loss_func(reward):
    
    reward_diff = reward[:, 0] - reward[:, 1]
    loss = -torch.log(torch.sigmoid(reward_diff)).mean()
    
    acc = torch.mean((reward_diff > 0).clone().detach().float())
    # print("loss", loss)
    # print("acc", acc)
    return loss, acc


if __name__ == "__main__":
    
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_seeds(args.seed + local_rank)
    
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(args.seed)
    
    train_dataset = S2C_Dataset("/data/liutao/mac8/json/train_91967.json")
    valid_dataset = S2C_Dataset("/data/liutao/mac8/json/val_11496.json")
    test_dataset = S2C_Dataset("/data/liutao/mac8/json/test_11496.json")

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    iters_per_epoch = int(math.ceil(len(train_dataset)*1.0/args.batch_size))
    # Set the training iterations.
    args.train_iters = args.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // args.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("train_dataset.iters_per_epoch = ", iters_per_epoch)
    print("len(train_loader) = ", len(train_loader))
    print("steps_per_valid = ", steps_per_valid)

    model = BlipReward(device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, args)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    if torch.distributed.get_rank() == 0:
        print("wandb init ……")
        wandb.init(config = args,
               project="blipreward",
               name = "first",
               job_type="training")

    # if get_rank() == 0:
    #     model.eval()
    #     valid_loss = []
    #     valid_acc_list = []
    #     with torch.no_grad():
    #         for step, batch_data_package in enumerate(valid_loader):
    #             reward = model(batch_data_package)
    #             loss, acc = loss_func(reward)
    #             valid_loss.append(loss)
    #             valid_acc_list.append(acc.item())

    #     # record valid and save best model
    #     print(valid_loss)
    #     valid_loss = torch.cat(valid_loss, 0)
    #     print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f' % (0, torch.mean(valid_loss), sum(valid_acc_list) / len(valid_acc_list)))

        # writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=0)
        # writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list), global_step=0)

    best_loss = 1e9
    optimizer.zero_grad()

    losses = []
    acc_list = []
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        for step, batch_data_package in enumerate(train_loader):
            model.train()
            
            reward = model(batch_data_package)
            loss, acc = loss_func(reward)
        
            # loss regularization
            loss = loss / args.gradient_accumulation_steps
            # back propagation
            loss.backward()

            losses.append(loss)
            acc_list.append(acc.item())

            iterations = epoch * len(train_loader) + step + 1
            train_iteration = iterations / args.gradient_accumulation_steps
            
            # update parameters of net
            if (iterations % args.gradient_accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # train result print and log 
                if get_rank() == 0:
                    # losses_log = torch.cat(losses, 0)
                    losses_log = torch.mean(torch.stack(losses))
                    print('Iteration %d | Loss %6.5f | Acc %6.4f' % (train_iteration, torch.mean(losses_log), sum(acc_list) / len(acc_list)))
                    wandb.log({'Train-Loss': torch.mean(losses_log),
                               'Train-Acc':sum(acc_list) / len(acc_list)})
                    # writer.add_scalar('Train-Loss', torch.mean(losses_log), global_step=train_iteration)
                    # writer.add_scalar('Train-Acc', sum(acc_list) / len(acc_list), global_step=train_iteration)
                    
                losses.clear()
                acc_list.clear()
            
            # valid result print and log
            if (iterations % steps_per_valid) == 0:
                if get_rank() == 0:
                    model.eval()
                    valid_loss = []
                    valid_acc_list = []
                    with torch.no_grad():
                        for step, batch_data_package in enumerate(valid_loader):
                            reward = model(batch_data_package)
                            # loss, loss_list, acc = loss_func(reward)
                            loss, acc = loss_func(reward)
                            valid_loss.append(loss)
                            valid_acc_list.append(acc.item())
                
                    # record valid and save best model
                    # valid_loss = torch.cat(valid_loss, 0)
                    valid_loss = torch.mean(torch.stack(valid_loss))
                    print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f' % (train_iteration, torch.mean(valid_loss), sum(valid_acc_list) / len(valid_acc_list)))

                    wandb.log({'Validation-Loss': torch.mean(valid_loss),
                    'Validation-Acc': sum(valid_acc_list) / len(valid_acc_list)})
                    # writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=train_iteration)
                    # writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list), global_step=train_iteration)
                        
                    if torch.mean(valid_loss) < best_loss:
                        print("Best Val loss so far. Saving model")
                        best_loss = torch.mean(valid_loss)
                        print("best_loss = ", best_loss)
                        save_model(model)
        




