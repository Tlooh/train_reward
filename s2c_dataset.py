import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
import clip
import math
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split
from utils import makedir



def _convert_image_to_rgb(image):
    return image.convert("RGB")


def image_transform(image_size):
    return Compose([
        Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(image_size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def split_train_val_test(json_file, ratios = [0.8, 0.1, 0.1], random_state=42):
    # 加载 JSON 数据
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 划分数据集
    train_data, temp_data = train_test_split(data, test_size=ratios[1] + ratios[2], random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=ratios[2] / (ratios[1] + ratios[2]), random_state=random_state)

    
    # 保存训练集、验证集、测试集为 JSON 文件
    save_json_dir = os.path.dirname(json_file)

    train_file_path = os.path.join(save_json_dir, f'train_{len(train_data)}.json')
    val_file_path = os.path.join(save_json_dir, f'val_{len(val_data)}.json')
    test_file_path = os.path.join(save_json_dir, f'test_{len(test_data)}.json')

    with open(train_file_path, 'w', encoding='utf-8') as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=4)

    with open(val_file_path, 'w', encoding='utf-8') as val_file:
        json.dump(val_data, val_file, ensure_ascii=False, indent=4)

    with open(test_file_path, 'w', encoding='utf-8') as test_file:
        json.dump(test_data, test_file, ensure_ascii=False, indent=4)

    # 打印信息
    print(f"Train data saved to: {train_file_path}")
    print(f"Validation data saved to: {val_file_path}")
    print(f"Test data saved to: {test_file_path}")

    return train_file_path, val_file_path, test_file_path



class S2C_Dataset(Dataset):
    def __init__(self, data_path, image_size = 224):
        self.tokenizer = init_tokenizer()
        self.preprocess = image_transform(image_size)

        # load json
        with open(data_path, "r") as f:
            self.data = json.load(f)
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # dict_item: img_worse, img_better, text_ids, text_mask
        dict_item = self.handle_data(self.data[index])
        return dict_item

    def store_dataset(self, dataset):
        makedir('/data/liutao/mac8/fixed_data')
        torch.save(self.data, os.path.join('/data/liutao/mac8/fixed_data', f"{dataset}.pth"))
        
    def handle_data(self, item):
        dict_item = {}
        simple_image_path = item['simple_img']
        complex_image_path = item['complex_img']

        simple_image = Image.open(simple_image_path)
        complex_image = Image.open(complex_image_path)
        simple_image = self.preprocess(simple_image)
        complex_image = self.preprocess(complex_image)

        text_input = self.tokenizer(item['simple'], padding = 'max_length', truncation=True, max_length = 35, return_tensors = "pt")

        dict_item["image_better"] = complex_image
        dict_item["image_worse"] = simple_image 
        dict_item["text_ids"] = text_input.input_ids
        dict_item["text_mask"] = text_input.attention_mask
            
        return dict_item



def make_dataset(train_json, val_json, test_json):
    
    train_dataset = S2C_Dataset(train_json, mode='make')
    train_dataset.store_dataset('train')
    print("train_data make finished!")

    val_dataset = S2C_Dataset(val_json, mode='make')
    val_dataset.store_dataset('valid')
    print("val_data make finished!")

    test_dataset = S2C_Dataset(test_json, mode='make')
    test_dataset.store_dataset('test')
    print("test_data make finished!")
    


class CLIP_Dataset(Dataset):
    def __init__(self, data_path, image_size = 224):
        self.preprocess = image_transform(image_size)

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

        # simple_image = Image.open(simple_image_path)
        # complex_image = Image.open(complex_image_path)
        # simple_image = self.preprocess(simple_image)
        # complex_image = self.preprocess(complex_image)

        dict_item["img_better"] = complex_image_path
        dict_item["img_worse"] = simple_image_path
        dict_item["text"] = item['simple']

        return dict_item
    

# # spilt train.json, val.json, test.json
# train_json, val_json, test_json = split_train_val_test("/data/liutao/mac8/json/S2C_114959.json")

# # make data
# make_dataset(train_json, val_json, test_json)




        
        
