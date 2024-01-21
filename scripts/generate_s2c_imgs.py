import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import json
import torch 
import argparse

import random
from PIL import Image
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import pdb

class S2C_Dataset(Dataset):
    def __init__(self, json_file):
        # 读取json文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本数据
        sample = self.data[idx]

        sample_item = {"id": sample['id']}
        sample_item["simple"] = sample["simple"]
        sample_item["complex"] = sample["complex"]

        return sample_item


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 创建成功")
    else:
        print(f"文件夹 {folder_path} 已经存在") 


def extract_index_dir(image_path):
    index = image_path.split('/')[-1]
    return int(index)


def img_is_exit(img_dir, img_name):
    img_files = os.listdir(img_dir)

    if img_name in img_files:
        return True
    
    return False

                
    

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# 4. schedule
scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

device = "cuda:3"
vae.to(device)
text_encoder.to(device)
unet.to(device)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# img_save_dir = "/media/sdb/liutao/datasets/rm_images/imgs"    

def run_inference(g, prompts, device):

    batch_size = len(prompts)

    # 1. get input_ids
    text_inputs = tokenizer(
        prompts, 
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    guidance_scale = 7.5
    do_classifier_free_guidance = guidance_scale > 1.0
    text_input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask.to(device)

    # 2.get prompt embedding
    prompt_embeds = text_encoder(
                text_input_ids.to(device),
                # attention_mask=None,
            )
    prompt_embeds = prompt_embeds[0]

    # do_classifier_free_guidance
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
    
    attention_mask = uncond_input.attention_mask.to(device)
    negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                # attention_mask=None,
            )
    negative_prompt_embeds = negative_prompt_embeds[0]
    

    # to avoid doing two forward passes
    # [8, 77, 768]
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare timesteps
    scheduler.set_timesteps(50, device=device)
    timesteps = scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = unet.config.in_channels
    shape = (batch_size, num_channels_latents, 64, 64)
    latents = torch.randn(shape, generator = g, device = device)
   
    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - 50 * scheduler.order

    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy() # (1, 512, 512, 3)
    images = (image * 255).round().astype("uint8")

    pil_images = [Image.fromarray(image) for image in images]

    return pil_images




def generate_images(args):
    # 1. init
    pattern = f"seed{args.seed}"
    # step_range = [0, 20000]
    # step_range = [20000, 40000]
    # step_range = [45000, 50000]
    # step_range = [50000, 55000]
    # step_range = [55000, 57500]
    # step_range = [57500, 60000]
    # step_range = [67500, 70000]
    step_range = [70000, 73000]
    step_start = step_range[0]
    step_end = step_range[1]

    progress_bar = tqdm(range(0, step_end - step_start))
    progress_bar.set_description("Generating Images……")


    # 2. load prompt dataset
    # 创建自定义数据集
    s2c_data = S2C_Dataset(args.prompt_json_path)
    print("Dataset的长度:", s2c_data.__len__(), "\t batch_size:", args.batch_size)
    # aplly  prompt pair (complex-simple) to generate images
    for step, batch in enumerate(s2c_data):
        
        if step < step_start:
            continue
        
        if step == step_end:
            print(f"{step_start} - {step_end} 生成结束!")
            break

        id = batch["id"]
        img_name = f"{id}.png"
        simple_img_save_path = os.path.join(args.img_save_dir, 'simple',img_name)
        complex_img_save_path = os.path.join(args.img_save_dir, 'complex',img_name)
        if os.path.exists(simple_img_save_path) and os.path.exists(complex_img_save_path):
            progress_bar.update(1)
            continue
        
        simple_text = batch['simple']
        complex_text = batch['complex']
        prompts = [simple_text, complex_text]

        g = torch.Generator(device=device).manual_seed(args.seed)
        images = run_inference(g, prompts, device)

        images[0].save(simple_img_save_path)
        images[1].save(complex_img_save_path)

        progress_bar.update(1)



if __name__ =="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data/liutao/user_lt/sd1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inferencing.")
    parser.add_argument(
        "--prompt_json_path",
        default="/home/khf/liutao/train_reward/json/S2C_114959.json",
        type=str,
        help="Path to the prompts json list file, each item of which is a dict with keys `id` and `prompt`.",
    )
    parser.add_argument(
        "--img_save_dir",
        default="/data/liutao/datasets/rm_images/images",
        type=str,
        help="Path to the generated images directory. The sub-level directory name should be the name of the model and should correspond to the name specified in `model`.",
    )
    parser.add_argument(
        "--model",
        default="default",
        type=str,
        help="""default(["sd1-4", "sd2-1"]), all or any specified model names splitted with comma(,).""",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size (per device) for the prompt dataloader.",
    )
    parser.add_argument(
        "--gpu_id",
        default=None,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        default=1,
        type=int,
        help="Num of images generated for each prompt.",
    )


    args = parser.parse_args()

    # create_folder_if_not_exists(args.img_save_dir)

    generate_images(args)
    # gen_from_last(args.img_save_dir)
    # extract_index_dir("/media/sdb/liutao/datasets/rm_images/images/6749")
            

