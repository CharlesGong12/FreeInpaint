import torch
import torch.nn as nn
import clip
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import transforms 
from PIL import Image
import yaml
import os
from munch import munchify
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
        config = munchify(config_dict)
    return config

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 256), 
        )
        self.last_layer = nn.Linear(256, 1, bias=False)
        self.last_layer_weight = self.last_layer.weight
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
        for name, param in self.last_layer.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        features = self.layers(input)
        out = self.last_layer(features)
        return out, features


class ViTBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, mlp_dim, dropout=0.1):
        super(ViTBlock, self).__init__()
        # Transformer encoder layer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True  # Input shape: (batch_size, seq_length, feature_dim)
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


class InpaintReward(nn.Module):
    def __init__(self, config_path, device='cuda:0', dtype=torch.float16):
        super().__init__()
        config = load_config(config_path)
        self.config = config
        self.device = device
        self.dtype = dtype
        
        self.clip_model, self.preprocess = clip.load(self.config['clip_path'], device=self.device)
        self.mlp = MLP(self.config['Reward']['mlp_dim']).to(self.device)
        self.vit_block = ViTBlock(self.config["ViT"]["feature_dim"], self.config["ViT"]["num_heads"], self.config["ViT"]["mlp_dim"]).to(self.device)
        # self.clip_model.to(dtype=self.dtype)
        self.mlp.to(dtype=self.dtype)
        self.vit_block.to(dtype=self.dtype)

        self.toImage = transforms.ToPILImage()

        self.mean = 0.4064   #0.65823
        self.std =  2.3021       #8.5400

        self.preprocess_pt = Compose([
        Resize((224, 224), interpolation=BICUBIC),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

        if self.config.fix_base:
            self.clip_model.requires_grad_(False)
    
    def score(
            self, 
            inpaint_image: Image.Image,
            masks_rgb: Image.Image
            ):
        
        inpaint_embeds_bs, mask_rgb_embeds_bs = [], []
 
        if isinstance(inpaint_image, torch.Tensor):
            inpaint = self.toImage(inpaint_image)
        else:
            inpaint = inpaint_image
        inpaint = self.preprocess(inpaint).unsqueeze(0)
        if isinstance(masks_rgb, torch.Tensor):
            mask_rgb = self.toImage(masks_rgb)
        else:
            mask_rgb = masks_rgb
        mask_rgb = self.preprocess(masks_rgb).unsqueeze(0)
        inpt, msk = inpaint.to(self.device), mask_rgb.to(self.device)
        
        inpt_embeds = self.clip_model.encode_image(inpt)
        msk_embeds = self.clip_model.encode_image(msk)

        inpaint_embeds_bs.append(inpt_embeds.squeeze(0))
        mask_rgb_embeds_bs.append(msk_embeds.squeeze(0))


        emb_inpaint = torch.stack(inpaint_embeds_bs, dim=0)
        emb_mask_rgb = torch.stack(mask_rgb_embeds_bs, dim=0)

        emb_feature = torch.cat((emb_inpaint, emb_mask_rgb), dim=-1)
        emb_feature = emb_feature.unsqueeze(1)
        emb_feature = self.vit_block(emb_feature) # 1024
      
        scores, last_features = self.mlp(emb_feature)
        scores = torch.squeeze(scores)
        last_features = torch.squeeze(last_features)

        if self.config.group:
            scores = (scores - self.mean) / self.std
    

        return scores.item(), last_features.detach().cpu()
    
    def __call__(self, inpaint_image: torch.Tensor, mask_rgb: torch.Tensor):
        img = self.preprocess_pt(inpaint_image).to(self.device, dtype=self.dtype)
        if mask_rgb.shape[1] == 1:
            mask_rgb = mask_rgb.repeat(1, 3, 1, 1)
        mask = self.preprocess_pt(mask_rgb).to(self.device, dtype=self.dtype)
        img_embed = self.clip_model.encode_image(img)
        mask_embed = self.clip_model.encode_image(mask)
        emb_feature = torch.cat((img_embed, mask_embed), dim=-1)
        emb_feature = emb_feature.unsqueeze(1)
        emb_feature = self.vit_block(emb_feature)
        scores, last_features = self.mlp(emb_feature)
        scores = torch.squeeze(scores)
        last_features = torch.squeeze(last_features)
        if self.config.group:
            scores = (scores - self.mean) / self.std
        return scores, last_features

    def load_model(self, model, ckpt_path = None):
        
        print('load checkpoint from %s'%ckpt_path)
        state_dict = {k: v for k, v in torch.load(ckpt_path, map_location='cpu').items()}
        new_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(new_dict)
        
        return model 