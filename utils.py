import os, shutil
import torch
import torch.distributed as dist
from config.options import *
import logging
from logging import handlers


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def make_path():
    return "{}_bs{}_fix={}_lr={}{}".format(args.savepath, args.BatchSize, args.fix_rate, args.lr, args.lr_decay_style)


def save_model(model):
    save_path = make_path()
    if not os.path.isdir(os.path.join(args.output_dir, save_path)):
        os.makedirs(os.path.join(args.output_dir, save_path), exist_ok=True)
    model_name = os.path.join(args.output_dir, save_path, 'best_lr={}.pt'.format(args.lr))
    torch.save(model.state_dict(), model_name)



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


"""===================== Tools ========================"""
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)


def find_missing_indices(folder_path, start_index, end_index):
    existing_indices = set()

    # 获取文件夹中的图像序号
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            index = int(filename.split(".")[0])
            existing_indices.add(index)

    # 找到缺失的图像序号
    missing_indices = []
    for i in range(start_index, end_index):
        if i not in existing_indices:
            missing_indices.append(i)

    return missing_indices


# # 用法示例
# folder_path = "/data/liutao/datasets/rm_images/images/complex"  # 替换为实际的文件夹路径
# start_index = 40000
# end_index = 80000

# missing_indices = find_missing_indices(folder_path, start_index, end_index)

# print("Missing indices:", missing_indices)
