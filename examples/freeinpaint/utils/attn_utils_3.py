import numpy as np
import torch
from examples.freeinpaint.utils.gaussian_smoothing_3 import GaussianSmoothing
from torch.nn import functional as F


def fn_get_topk(attention_map, K=1):
    H, W = attention_map.size()
    attention_map_detach = attention_map.view(H * W)  # .detach()
    topk_value, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []
    topk_value_list = []
    for i in range(len(topk_index)):
        index = topk_index[i].cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
        topk_value_list.append(topk_value[i])
    return topk_coord_list, topk_value_list


def fn_get_topk_plus(attention_map, K=1, threshold=0.8):
    H, W = attention_map.size()
    attention_map_detach = attention_map.view(H * W)  # .detach()
    topk_value, topk_index = attention_map_detach.topk(H * W, dim=0, largest=True, sorted=True)
    topk_coord_list = []
    topk_value_list = []
    threshold_coord_list = []
    threshold_value_list = []
    for i in range(len(topk_index)):
        index = topk_index[i].cpu().numpy()
        coord = index // W, index % W
        if i < K:  # topk_value[i]> threshold and
            topk_coord_list.append(coord)
            topk_value_list.append(topk_value[i])
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[i])
        elif i < 4 * K or topk_value[i] > threshold:
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[i])
        if i > 4 * K and topk_value[i] < threshold: break
    """if len(topk_value_list)==0:
            index = topk_index[0].cpu().numpy()
            coord = index // W, index % W
            topk_coord_list.append(coord)
            topk_value_list.append(topk_value[0])
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[0])"""

    return topk_coord_list, topk_value_list, threshold_coord_list, threshold_value_list


def fn_smoothing_func(attention_map):
    smoothing = GaussianSmoothing().to(attention_map.device)
    if len(attention_map.size()) == 2:
        attention_map = F.pad(attention_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
        attention_map = smoothing(attention_map).squeeze(0).squeeze(0)
    elif len(attention_map.size()) == 3:
        smoothed_attention_maps = []
        for i in range(attention_map.size(2)):
            attention_map_i = F.pad(attention_map[:, :, i].unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
            smoothed_attention_i = smoothing(attention_map_i).squeeze(0).squeeze(0)
            smoothed_attention_maps.append(smoothed_attention_i)

        attention_map = torch.stack(smoothed_attention_maps, dim=2)  # 将列表转换为张量
    
    return attention_map


def fn_show_attention(
        cross_attention_maps,
        self_attention_maps,
        indices,
        K=1,
        attention_res=16,
        smooth_attentions=True,
):
    cross_attention_map_list, self_attention_map_list = [], []

    # cross attention map preprocessing
    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]

    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
        cross_attention_map_list.append(cross_attention_map_per_token)

    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        topk_coord_list, topk_value = fn_get_topk(cross_attention_map_per_token, K=K)

        self_attention_map_per_token_list = []
        for coord_x, coord_y in topk_coord_list:
            self_attention_map_per_token = self_attention_maps[coord_x, coord_y]
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
            self_attention_map_per_token_list.append(self_attention_map_per_token)

        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / len(
                self_attention_map_per_token_list)
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
                                            (
                                                        self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)

        self_attention_map_list.append(norm_self_attention_map_per_token)

    # tensor to numpy
    cross_attention_map_numpy = torch.cat(cross_attention_map_list, dim=0).cpu().detach().numpy()
    self_attention_map_numpy = torch.cat(self_attention_map_list, dim=0).cpu().detach().numpy()

    return cross_attention_map_numpy, self_attention_map_numpy


def fn_show_attention_plus(
        cross_attention_maps,
        self_attention_maps,
        indices,
        K=1,
        attention_res=16,
        smooth_attentions=True,
):
    # cross attention map preprocessing
    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]
    cross_attention_map_list = []
    cross_attention_map_list_for_show = []
    otsu_masks = []
    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
        topk_coord_list, topk_value_list = fn_get_topk(cross_attention_map_per_token, K=K)
        cross_attention_map_list_for_show.append(cross_attention_map_per_token.to(torch.float16).cpu().detach().numpy())
        cross_attention_map_list.append(cross_attention_map_per_token)

        # -----------------------------------
        # clean cross_attention_map_cur_token
        # -----------------------------------
        clean_cross_attention_map_per_token_mask = fn_get_otsu_mask(cross_attention_map_per_token)
        # clean_cross_attention_map_per_token_mask = fn_clean_mask(clean_cross_attention_map_per_token_mask, topk_coord_list[0][0], topk_coord_list[0][1])
        otsu_masks.append(clean_cross_attention_map_per_token_mask)

    self_attention_map_list = []
    self_attention_map_list_list=[]
    for i in range(len(cross_attention_map_list)):
        cross_attn_map_cur_token = cross_attention_map_list[i]
        self_attn_map_cur_token = torch.zeros_like(cross_attn_map_cur_token)
        mask_cur_token = otsu_masks[i]
        cross_attn_value_cur_token_sum = 0
        self_attention_map_per_token_list = []
        self_atten_map_list=[]
        for j in range(attention_res):
            for k in range(attention_res):
                if mask_cur_token[j, k] == 0: continue
                cross_attn_value_cur_token = cross_attn_map_cur_token[j, k]
                cross_attn_value_cur_token_sum = cross_attn_value_cur_token_sum + cross_attn_value_cur_token
                self_attn_map_cur_position = self_attention_maps[j, k].view(attention_res, attention_res).contiguous()
                self_attn_map_cur_token = self_attn_map_cur_token + cross_attn_value_cur_token * self_attn_map_cur_position
                self_atten_map_list.append(self_attn_map_cur_position)
        self_attention_map_per_token_list.append(self_attn_map_cur_token)
        self_attention_map_list_list.append(self_atten_map_list)
        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / cross_attn_value_cur_token_sum
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
                                            (
                                                    self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        # print(norm_self_attention_map_per_token.shape)
        self_attention_map_list.append(norm_self_attention_map_per_token.to(torch.float16).cpu().detach().numpy())

    # tensor to numpy
    # cross_attention_map_numpy = torch.cat(cross_attention_map_list, dim=0).cpu().detach().numpy()
    # self_attention_map_numpy = torch.cat(self_attention_map_list, dim=0).cpu().detach().numpy()

    return cross_attention_map_list_for_show, self_attention_map_list  #, self_attention_map_list_list


import cv2


def fn_get_otsu_mask(x):
    x_numpy = x.to(torch.float16)
    x_numpy = x_numpy.cpu().detach().numpy()
    x_numpy = x_numpy * 255
    x_numpy = x_numpy.astype(np.uint16)

    opencv_threshold, _ = cv2.threshold(x_numpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opencv_threshold = opencv_threshold * 1. / 255.

    otsu_mask = torch.where(
        x < opencv_threshold,
        torch.tensor(0, dtype=x.dtype, device=x.device),
        torch.tensor(1, dtype=x.dtype, device=x.device))

    return otsu_mask


def fn_clean_mask(otsu_mask, x, y):
    H, W = otsu_mask.size()
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def dfs(cur_x, cur_y):
        if cur_x >= 0 and cur_x < H and cur_y >= 0 and cur_y < W and otsu_mask[cur_x, cur_y] == 1:
            otsu_mask[cur_x, cur_y] = 2
            for delta_x, delta_y in direction:
                dfs(cur_x + delta_x, cur_y + delta_y)

    dfs(x, y)
    ret_otsu_mask = torch.where(
        otsu_mask < 2,
        torch.tensor(0, dtype=otsu_mask.dtype, device=otsu_mask.device),
        torch.tensor(1, dtype=otsu_mask.dtype, device=otsu_mask.device))

    return ret_otsu_mask
