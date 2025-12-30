import torch
import cv2
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from urllib.request import urlretrieve 
from PIL import Image
import open_clip
import os
import hpsv2
import ImageReward as RM

import copy
import matplotlib.pyplot as plt
# from daam import trace
import random
random.seed(1)
import pandas as pd

from examples.freeinpaint.metrics import get_mask_bbox
from examples.freeinpaint.metrics.prefpaint import InpaintReward
import requests

# Run-Length Encoding, RLE
def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

def mask2convexhull(mask):
    mask = np.array(mask)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hull = cv2.convexHull(contours[0])
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [convex_hull], 1)
    return mask

def apply_dilation(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def generate_masks_with_varying_precision(mask, s, S, kernel_size_range):
    if s == 0:
        return mask
    else:
        if mask.shape[-1] == 1:
            mask = mask[:,:,0]
        x_indices, y_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return mask  
        
        xmin, xmax = min(x_indices), max(x_indices)
        ymin, ymax = min(y_indices), max(y_indices)

        kernel_size = int(kernel_size_range[0] + (kernel_size_range[1] - kernel_size_range[0]) * (s / S))
        dilated_mask = apply_dilation(mask, kernel_size)

        bbox_mask = np.zeros_like(mask)
        bbox_mask[xmin:xmax+1, ymin:ymax+1] = dilated_mask[xmin:xmax+1, ymin:ymax+1]

        return bbox_mask[:,:,np.newaxis]

def generate_bbox_mask(mask):
    if mask.shape[-1] == 1:
        mask = mask[:,:,0]
    x_indices, y_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return mask

    xmin, xmax = min(x_indices), max(x_indices)
    ymin, ymax = min(y_indices), max(y_indices)

    bbox_mask = np.zeros_like(mask)
    bbox_mask[xmin:xmax+1, ymin:ymax+1] = 1

    x, y, w, h = cv2.boundingRect(bbox_mask)
    random_shape_mask = np.zeros_like(bbox_mask)
    expansion_factor = 0.08
    x_min = x - int(w * expansion_factor)
    y_min = y - int(h * expansion_factor)
    x_max = x + w + int(w * expansion_factor)
    y_max = y + h + int(h * expansion_factor)
    
    num_vertices = 30
    # seed has been set to 1
    points = [(random.randint(x_min, x_max), random.randint(y_min, y_max)) for _ in range(num_vertices)]
    hull = cv2.convexHull(np.array(points))
    
    cv2.fillConvexPoly(random_shape_mask, hull, 1)
    mask = np.clip(bbox_mask + random_shape_mask, 0, 1)

    return mask[:,:,np.newaxis]


class MetricsCalculator:
    def __init__(
            self, 
            device,
            clip_metric_model_name_or_path,
            image_reward_model_name_or_path,
            inpaint_reward_config_path,
            inpaint_reward_model_path,
            ) -> None:
        self.device=device
        # clip
        self.clip_metric_calculator = CLIPScore(model_name_or_path=clip_metric_model_name_or_path).to(device)
        # lpips
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        # image reward model
        self.imagereward_model = RM.load("ImageReward-v1.0", download_root=image_reward_model_name_or_path, device=device)
        # inpaint reward
        self.inpaint_reward = InpaintReward(inpaint_reward_config_path, device=device)
        self.inpaint_reward = self.inpaint_reward.load_model(self.inpaint_reward, inpaint_reward_model_path)
 

    def calculate_image_reward(self,image,prompt):
        reward = self.imagereward_model.score(prompt, [image])
        return reward

    def calculate_hpsv21_score(self,image,prompt):
        result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
        return result.item()

    def calculate_clip_similarity(self, img, txt, mask=None):
        img = np.array(img)

        if mask is not None:
            mask = np.array(mask)
            x_min, y_min, x_max, y_max = get_mask_bbox(mask)
            img = img[y_min:y_max, x_min:x_max]
        
        img_tensor=torch.tensor(img).permute(2,0,1).to(self.device)
        
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        
        return score

    
    def calculate_lpips(self, img_gt, img_pred, mask=None):
        if img_gt.size != img_pred.size:
            img_gt = img_gt.resize(img_pred.size)
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_gt.shape == img_pred.shape, f"Image shapes should be the same, but got {img_gt.shape} and {img_pred.shape}."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask 
            img_gt = img_gt * mask
            # crop the image to the mask
            x_min, y_min, x_max, y_max = get_mask_bbox(mask)
            img_pred = img_pred[y_min:y_max, x_min:x_max]
            img_gt = img_gt[y_min:y_max, x_min:x_max]
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
        
        try:
            score =  self.lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
            score = score.cpu().item()
        except:
            # set None to score if the calculation failed
            score = None
        return score
    
    def calculate_inpaint_reward(self, img, mask):
        mask = Image.fromarray((mask[:,:,0]*255).astype(np.uint8))
        score, _ = self.inpaint_reward.score(img, mask)
        return score