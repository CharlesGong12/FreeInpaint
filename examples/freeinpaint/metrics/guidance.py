import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import numpy as np
from PIL import Image
import os

import ImageReward as RM
from transformers import CLIPTokenizer, CLIPModel, CLIPImageProcessor


class ImageRewardScore:
    def __init__(self, model_name_or_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.model = RM.load("ImageReward-v1.0", download_root=model_name_or_path)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.model.to(device, dtype=dtype)
        self.transform = Compose([
                            Resize((224, 224), interpolation=BICUBIC),
                            CenterCrop((224, 224)),
                            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                        ])
    
    def process_text(self, text):
        text_input = self.model.blip.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
        return text_input

    def __call__(self, text_input, image):     
        image = self.transform(image).to(self.device, dtype=self.dtype)     
        image_embeds = self.model.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.model.blip.text_encoder(text_input.input_ids.to(self.device),
                                                    attention_mask = text_input.attention_mask.to(self.device),
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                )
        
        txt_features = text_output.last_hidden_state[:,0,:] # (feature_dim)
        rewards = self.model.mlp(txt_features)
        rewards = (rewards - self.model.mean) / self.model.std
        
        return rewards
    
def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def get_mask_bbox_pt(mask: torch.Tensor) -> torch.Tensor:
    if len(mask.shape) == 3:
        mask = torch.sum(mask, axis=0)
    xm = torch.nonzero(torch.sum(mask, axis=0) > 0).squeeze()
    ym = torch.nonzero(torch.sum(mask, axis=1) > 0).squeeze()
    try:
        x_min, x_max = xm.min(), xm.max() + 1
    except:
        x_min, x_max = 0, mask.shape[1]
    try:
        y_min, y_max = ym.min(), ym.max() + 1
    except:
        y_min, y_max = 0, mask.shape[0]
    return torch.tensor([x_min, y_min, x_max, y_max])
    
class PromptRewardScore:
    def __init__(self, model_name_or_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device, dtype=dtype)
        self.transform = Compose([
                            Resize((224, 224), interpolation=BICUBIC),
                            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                        ])
        self.device = device
        self.dtype = dtype

    def process_text(self, text):
        text_input = self.tokenizer(
            text, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length,
            truncation=True, 
            return_tensors="pt"
            ).input_ids.to(self.device)
        text_embeds = self.model.get_text_features(text_input)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

    def __call__(
            self, 
            text_embeds,
            img,
            mask=None,
            ):

        if mask is not None:
            img = img * mask

        img = self.transform(img).to(self.device, dtype=self.dtype)
        
        img_embeds = self.model.get_image_features(img)
        img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

        return -spherical_dist_loss(text_embeds, img_embeds).mean()
