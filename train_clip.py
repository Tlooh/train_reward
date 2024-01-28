import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import sys
import argparse
import json
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
import wandb
import logging
from logging import handlers

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from clip_reward import CLIPReward
from tqdm import tqdm
from scheduler import *

import pdb



"""====================== CLIP Dataset ======================"""

class CLIP_Dataset(Dataset):
    def __init__(self, data_path):
        # load json
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # dict_item: img_worse, img_better, text_ids, text_mask
        dict_item = self.handle_data(self.data[index])
        return dict_item
        
    def handle_data(self, item):
        dict_item = {}
        simple_image_path = item['simple_img']
        complex_image_path = item['complex_img']

        dict_item["img_better"] = complex_image_path
        dict_item["img_worse"] = simple_image_path
        dict_item["text"] = item['simple']

        return dict_item


"""===================== logging ========================"""
def logger_config(log_path, level=logging.INFO,fmt='%(asctime)s | %(levelname)s: %(message)s'):
    logger = logging.getLogger(log_path)
    format_str = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M') #设置日志格式
    logger.setLevel(level = level) #设置日志级别
    console = logging.StreamHandler() #往屏幕上输出
    console.setFormatter(format_str) #设置屏幕上显示的格式
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setFormatter(format_str)#设置文件里写入的格式
    logger.addHandler(console) #把对象加到logger里
    logger.addHandler(handler)

    return logger
    

"""====================== Training ======================"""
class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ['linear', 'cosine', 'exponential', 'constant', "inverse_square_root", 'None']

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style=None, last_iter=-1, decay_ratio=0.5):
        assert warmup_iter <= num_iters
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        self.decay_ratio = decay_ratio
        self.step(self.num_iters)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f'learning rate decaying style {self.decay_style}, ratio {self.decay_ratio}')

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        if self.decay_style == "inverse_square_root":
            return self.start_lr * math.sqrt(self.warmup_iter) / math.sqrt(max(self.warmup_iter, self.num_iters))
        elif self.decay_style == "constant":
            return self.start_lr
        else:
            if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
                return float(self.start_lr) * self.num_iters / self.warmup_iter
            else:
                if self.decay_style == "linear":
                    decay_step_ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                    return self.start_lr - self.start_lr * (1 - self.decay_ratio) * decay_step_ratio
                elif self.decay_style == "cosine":
                    decay_step_ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                    return self.start_lr * (
                            (math.cos(math.pi * decay_step_ratio) + 1) / 2 * (1 - self.decay_ratio) + self.decay_ratio)
                elif self.decay_style == "exponential":
                    # TODO: implement exponential decay
                    raise NotImplementedError
                else:
                    raise NotImplementedError

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        sd = {
            # 'start_lr': self.start_lr,
            'warmup_iter': self.warmup_iter,
            'num_iters': self.num_iters,
            'decay_style': self.decay_style,
            'end_iter': self.end_iter,
            'decay_ratio': self.decay_ratio
        }
        return sd

    def load_state_dict(self, sd):
        # self.start_lr = sd['start_lr']
        self.warmup_iter = sd['warmup_iter']
        self.num_iters = sd['num_iters']
        self.end_iter = sd['end_iter']
        self.decay_style = sd['decay_style']
        if 'decay_ratio' in sd:
            self.decay_ratio = sd['decay_ratio']
        self.step(self.num_iters)

    def switch_linear(self, args):
        current_lr = self.get_lr()
        self.start_lr = current_lr
        self.end_iter = args.num_epochs - self.num_iters
        self.num_iters = 0
        self.decay_style = "linear"


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    args.gradient_accumulation_steps = 1
    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters // args.gradient_accumulation_steps
    
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters - warmup_iter,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio)

    return lr_scheduler


def loss_sig_fn(reward):
    reward_diff = reward[:, 0] - reward[:, 1]
    loss = -torch.log(torch.sigmoid(reward_diff)).mean()
    acc = torch.mean((reward_diff > 0).clone().detach().float())
    return loss, acc

def loss_nce_fn(reward):
    
    # print(reward)
    target = torch.ones(reward.shape[0], dtype=torch.long).to(reward.device)
    loss_list = F.cross_entropy(reward, target, reduction='none')
    # print(loss_list)
    loss = torch.mean(loss_list)

    reward_diff = reward[:, 0] - reward[:, 1]
    acc = torch.mean((reward_diff > 0).clone().detach().float())

    return loss, acc



def trainer(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    
    # load dataset
    train_dataset = CLIP_Dataset("/data/liutao/mac8/json/train_91967.json")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = CLIP_Dataset("/data/liutao/mac8/json/val_11496.json")
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    # load model
    model = CLIPReward(clip_name=args.CLIP_NAME, device=device)
    model.to(device)
    model.train()

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    args.wd = 0.35
    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    # load optimizer, scheduler
    args.train_iters = args.num_epochs * len(train_dataloader)
    # optimizer = torch.optim.Adam(
    #     [
    #     {"params":model.clip_model.visual.parameters(), "lr": args.lr * 0.1},
    #     {"params":model.mlp.parameters()},

    # ], lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, args)
    # scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.train_iters)

    best_loss = 1e9
    best_acc = 0
    global_step = 0
    for epoch in range(args.num_epochs):
        loss_list = []
        acc_list = []
        args.log.info(f"===================== Epoch {epoch + 1} / {args.num_epochs} =====================")
        model.train()
        for step, batch_data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
            reward = model(batch_data)
            if args.loss_fn == "nce":
                loss, acc = loss_nce_fn(reward)
            else:
                loss, acc = loss_sig_fn(reward)

            # back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # record metric
            loss_list.append(loss.item())
            acc_list.append(acc.item())

            global_step += 1
            avg_loss = sum(loss_list) // len(loss_list)
            avg_acc = sum(acc_list) // len(acc_list)
            args.log.info(f"Step {global_step} | loss: {loss.item():.4f} | acc: {acc.item():.4f}")

            if args.wandb:
                wandb.log({'Train-Loss': avg_loss,
                            'Train-Acc': avg_acc})
            

        # model.eval()
        # valid_loss_list = []
        # valid_acc_list = []
        # with torch.no_grad():
        #     for step, batch_data in enumerate(valid_loader):
        #         reward = model(batch_data)
        #         loss, acc = loss_nce_fn(reward)
        #         valid_loss_list.append(loss.item())
        #         valid_acc_list.append(acc.item())
        
        # valid_avg_loss = sum(valid_loss_list) / len(valid_loss_list)
        # valid_avg_acc = sum(valid_acc_list) / len(valid_acc_list)
        # args.log.info(f"Valid | avg_loss: {valid_avg_loss:.4f} | avg_acc: {valid_avg_acc:.4f}")

        # if args.wandb:
        #     wandb.log({'Validation-Loss': valid_avg_loss,
        #     'Validation-Acc': valid_avg_acc})

        # if valid_avg_acc > best_acc:
        #     args.log.info("Best Val acc so far. Saving model")
        #     best_acc = valid_avg_acc
        #     best_loss = valid_avg_loss
        #     args.log.info(f"Best acc: {valid_avg_acc:.4f}")
        #     save_model(model, args)

            # do validate
            if global_step % args.valid_per_epoch == 0 or (global_step + 1) == args.train_iters:
                model.eval()
                valid_loss_list = []
                valid_acc_list = []
                with torch.no_grad():
                    for step, batch_data in enumerate(valid_loader):
                        reward = model(batch_data)
                        if args.loss_fn == "nce":
                            loss, acc = loss_nce_fn(reward)
                        else:
                            loss, acc = loss_sig_fn(reward)
                        valid_loss_list.append(loss.item())
                        valid_acc_list.append(acc.item())
                
                valid_avg_loss = sum(valid_loss_list) / len(valid_loss_list)
                valid_avg_acc = sum(valid_acc_list) / len(valid_acc_list)
                args.log.info(f"Valid | avg_loss: {valid_avg_loss:.4f} | avg_acc: {valid_avg_acc:.4f}")

                if args.wandb:
                    wandb.log({'Validation-Loss': valid_avg_loss,
                    'Validation-Acc': valid_avg_acc})

                if valid_avg_acc > best_acc:
                    args.log.info("Best Val acc so far. Saving model")
                    best_acc = valid_avg_acc
                    best_loss = valid_avg_loss
                    args.log.info(f"Best acc: {valid_avg_acc:.4f}")
                    save_model(model, args)


"""====================== Basic setting ======================"""

def save_model(model, args):
    model_name = f"bs{args.batch_size}_lr={args.lr}_{args.loss_fn}.pt"
    model_path = os.path.join(args.save_dir, model_name)
    torch.save(model.state_dict(), model_path)



def main():
    parser = argparse.ArgumentParser(description="Training CLIP Reward")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_dir", type=str, default="/data/liutao/checkpoints/ClipReward", help="Directory to save models")
    parser.add_argument("--CLIP_NAME", type=str, default="ViT-L/14", help="GPU ID")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
    parser.add_argument('--warmup', type=float, default=0.0,
                    help='percentage of data to warmup on (.01 = 1% of all '
                        'training iters). Default 0.01')
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.98)
    parser.add_argument('--adam-eps', type=float, default=1e-6)
    parser.add_argument('--lr-decay-iters', type=int, default=None,
                    help='number of iterations to decay LR over,'
                        ' If None defaults to `--train-iters`*`--epochs`')
    parser.add_argument('--lr-decay-style', type=str, default='cosine',
                        choices=['constant', 'linear', 'cosine', 'exponential', 'inverse_square_root'],
                        help='learning rate decay function')
    parser.add_argument('--lr-decay-ratio', type=float, default=0.0)
    parser.add_argument('--valid_per_epoch', type=int, default=50)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--loss_fn', type=str, default="nce")

    args = parser.parse_args()

    log_path = os.path.join(args.save_dir, "logs",'log_CLIP.log')
    if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
    args.log = logger_config(log_path)

    if args.wandb:
        print("wandb init ……")
        wandb.init(config = args,
               project="CLIP_Reward",
               name = "first",
               job_type="training")

    trainer(args)




if __name__ == "__main__":
    main()