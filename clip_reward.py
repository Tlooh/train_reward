import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import clip

CLIP_DIMS = {"ViT-L/14":768,}

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        # for name, param in self.layers.named_parameters():
        #     if 'weight' in name:
        #         nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
        #     if 'bias' in name:
        #         nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)




class CLIPReward(nn.Module):
    def __init__(self, clip_name = "ViT-L/14", device = 'cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(clip_name, device = device)
        self.mlp = MLP(input_size=CLIP_DIMS[clip_name])
        # self.linear = nn.Linear(CLIP_DIMS[clip_name], 1)

        self.clip_model.logit_scale.requires_grad_(False)

    def forward(self, batch_data):
        # encode data
        batch_data = self.encode_pair(batch_data)
        # forward
        emb_better, emb_worse = batch_data['emb_better'], batch_data['emb_worse']

        reward_better = self.mlp(emb_better)
        reward_worse = self.mlp(emb_worse)

        reward = torch.concat((reward_better, reward_worse), dim=1)

        return reward



    def encode_pair(self, batch_data):
        # images：path
        img_better, img_worse , text= batch_data['img_better'], batch_data['img_worse'], batch_data['text']

        img_better  = [self.preprocess(Image.open(img)) for img in img_better ]
        img_better = torch.stack(img_better).to(self.device) # [bsz, 3, 224, 224]
        img_worse  = [self.preprocess(Image.open(img)) for img in img_worse ]
        img_worse = torch.stack(img_worse).to(self.device) # [bsz, 3, 224, 224]

        emb_better = F.normalize(self.clip_model.encode_image(img_better)) # [bsz, 768]
        emb_worse = F.normalize(self.clip_model.encode_image(img_worse)) # [bsz, 768]

        # text = clip.tokenize(text, truncate=True).to(self.device) # [1, 77]
        # text_features = F.normalize(self.clip_model.encode_text(text)) # [1, 768]

        
        # logit_scale = self.clip_model.logit_scale

        # similarity = logit_scale * emb_better @ text_features.T
        # print(similarity)
        # print(similarity.mean())
        
        # get batch data
        batch_data = {
            'emb_better': emb_better.float(),
            'emb_worse': emb_worse.float(),
        }


        return batch_data
        


    def score(self, prompt, image):
        # support image_path:str or image:Image
        if isinstance(image, str):
            image_path = image
            pil_image = Image.open(image_path)
        elif isinstance(image, Image.Image):
            pil_image = image

        text = clip.tokenize(prompt, truncate=True).to(self.device) # [1, 77]
        text_features = F.normalize(self.model.encode_text(text)) # [1, 768]

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device) # [1, 3, 224, 224]
        image_features = F.normalize(self.model.encode_image(image)) # [1, 768]
        logit_scale = self.model.logit_scale.exp()

        similarity = logit_scale * image_features @ text_features.T

        return similarity.detach().cpu().numpy().item()



    # def forward(self, image, text):
    #     image_features = self.encode_image(image)
    #     text_features = self.encode_text(text)

    #     # normalized features
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)

    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()

    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_image, logits_per_text