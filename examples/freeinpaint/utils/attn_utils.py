import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from PIL import Image

from examples.freeinpaint.utils.gaussian_smoothing import GaussianSmoothing


def fn_get_topk(attention_map, K=1):
    H, W = attention_map.size()
    attention_map_detach = attention_map.detach().view(H * W)
    topk_value, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []

    for index in topk_index:
        index = index.cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
    return topk_coord_list, topk_value


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
        # for i in range(attention_map.size(2)):
        #     attention_map_i = F.pad(attention_map[:, :, i].unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
        #     attention_map[:, :, i] = smoothing(attention_map_i).squeeze(0).squeeze(0)

    return attention_map

def fn_show_attention_inpaint(
    cross_attention_maps,
    self_attention_maps,
    attention_mask,
    mask,
    attention_res=(16, 16),
    smooth_attentions=True,
):
    mask = torch.nn.functional.interpolate(mask, size=attention_res, mode='bicubic').squeeze(0)   # [1, res, res]
    mask = mask[0].unsqueeze(2)   # [res, res, 1]

    # cross attention map preprocessing
    cross_attention_maps = cross_attention_maps[:, :, 1:]
    attention_mask = attention_mask[:, 1:] # [1, num_tokens]
    nonpad_len = attention_mask.sum() - 1
    cross_attention_maps = cross_attention_maps[:, :, :nonpad_len]  # [res, res, num_tokens]

    # Shift indices since we removed the first token
    # indices = [index - 1 for index in indices]
    # if indices is None:
    #     indices = range(cross_attention_maps.size(2))

    # for i in indices:
    #     cross_attention_map_per_token = cross_attention_maps[:, :, i]
    #     if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
    #     cross_attention_map_list.append(cross_attention_map_per_token)
    cross_attention_map = cross_attention_maps.mean(dim=2)  # [res, res]
    if smooth_attentions: cross_attention_map = fn_smoothing_func(cross_attention_map)
    cross_attention_map = (cross_attention_map - cross_attention_map.min()) / \
        (cross_attention_map.max() - cross_attention_map.min() + 1e-8)

    # for i in indices:
    #     cross_attention_map_per_token = cross_attention_maps[:, :, i]
    #     topk_coord_list, topk_value = fn_get_topk(cross_attention_map_per_token, K=K)

    #     self_attention_map_per_token_list = []
    #     for coord_x, coord_y in topk_coord_list:

    #         self_attention_map_per_token = self_attention_maps[coord_x, coord_y]
    #         self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
    #         self_attention_map_per_token_list.append(self_attention_map_per_token)

    #     if len(self_attention_map_per_token_list) > 0:
    #         self_attention_map_per_token = sum(self_attention_map_per_token_list) / len(self_attention_map_per_token_list)
    #         if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
    #     else:
    #         self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
    #         self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

    #     norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
    #         (self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        
    #     self_attention_map_list.append(norm_self_attention_map_per_token)

    mask_indices = (mask == 1).nonzero(as_tuple=True)  # 获取mask=1的像素的索引
    # mask_indices: ([行索引], [列索引])
    if len(mask_indices[0]) > 0: # 保证mask中存在1的区域，避免空Tensor导致错误
        masked_attention_maps = self_attention_maps[mask_indices[0], mask_indices[1]]  # [num_masked_pixels, res*res]
        masked_attention_maps = masked_attention_maps.reshape(-1, attention_res[0], attention_res[1]) # [num_masked_pixels, res, res]

        self_attention_map = masked_attention_maps.mean(dim=0)  # [res, res]

        if smooth_attentions:
            self_attention_map = fn_smoothing_func(self_attention_map)
        self_attention_map = (self_attention_map - self_attention_map.min()) / \
            (self_attention_map.max() - self_attention_map.min() + 1e-8)
    else:
        self_attention_map = torch.zeros((attention_res, attention_res), device=self_attention_maps.device)  # 如果mask全是0, 返回全0的attention map

    # tensor to numpy
    cross_attention_map_numpy       = cross_attention_map.cpu().detach().numpy()
    self_attention_map_numpy        = self_attention_map.cpu().detach().numpy() 

    return cross_attention_map_numpy, self_attention_map_numpy

def visualize_and_save_attention(
    cross_attention_map_numpy_list,
    self_attention_map_numpy_list,
    image,
    crop=False,
    save_path=".",
    filename_prefix="attn",
    cmap='jet'# 'OrRd',
):
    """
    可视化并保存 attention map，不显示坐标轴。

    Args:
        cross_attention_map_numpy_list: 包含多个 cross-attention map 的列表，每个 map 的维度是 [res, res]。
        self_attention_map_numpy_list: 包含多个 self-attention map 的列表，每个 map 的维度是 [res, res]。
        save_path: 保存图片的路径。
        filename_prefix: 文件名前缀。
        cmap: matplotlib colormap.
    """

    # 确保 save_path 存在
    os.makedirs(save_path, exist_ok=True)

    # 计算平均 attention map
    cross_attention_map_numpy = sum(cross_attention_map_numpy_list) / len(
        cross_attention_map_numpy_list
    )
    self_attention_map_numpy = sum(self_attention_map_numpy_list) / len(
        self_attention_map_numpy_list
    )
    cross_attention_map_numpy_1000 = cross_attention_map_numpy_list[0]
    self_attention_map_numpy_1000 = self_attention_map_numpy_list[0]

    # # 定义可视化函数，避免重复代码
    # def plot_and_save(data, filename, title):
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)  # 设置颜色范围
    #     ax.axis("off")  # 关闭坐标轴
    #     # fig.colorbar(im, ax=ax, shrink=0.8)
    #     # ax.set_title(title)
    #     # 移除空白边缘
    #     plt.savefig(
    #         os.path.join(save_path, filename), bbox_inches="tight", pad_inches=0
    #     )
    #     plt.close(fig)
    # 加载原始图像
    image = np.array(image)
    image = image / 255.0  # 归一化到 [0, 1]

    # 定义可视化和保存函数
    def plot_and_save(data, filename, title, crop=False):
        data = data.astype(np.float32)
        # 将注意力图缩放到图像大小
        if crop:
            data = np.rot90(data, k=1)
        attention_map_resized = np.array(Image.fromarray(data).resize((image.shape[1], image.shape[0]), Image.LINEAR))

        # 确保注意力图的数值范围在 0 到 1 之间
        attention_map_resized = np.clip(attention_map_resized, 0, 1)

        # 将注意力图转换为颜色图
        cmap_obj = plt.get_cmap(cmap)
        attention_map_colored = cmap_obj(attention_map_resized)[:, :, :3]  # 获取 RGB 值

        # Blend 图像和注意力图
        blended_image = image * 0.5 + attention_map_colored * 0.5

        # 确保 blend 后的图像数值范围在 0 到 1 之间
        blended_image = np.clip(blended_image, 0, 1)


        fig, ax = plt.subplots()
        ax.imshow(blended_image)
        ax.axis("off")  # 关闭坐标轴

        # 移除空白边缘并保存
        plt.savefig(
            os.path.join(save_path, filename), bbox_inches="tight", pad_inches=0
        )
        plt.close(fig)

    # 可视化并保存 cross-attention map
    plot_and_save(
        cross_attention_map_numpy,
        f"{filename_prefix}_cross_attn.png",
        "Cross-Attention Map (Average)",
        crop=crop,
    )

    # 可视化并保存 self-attention map
    plot_and_save(
        self_attention_map_numpy,
        f"{filename_prefix}_self_attn.png",
        "Self-Attention Map (Average)",
        crop=crop,
    )

    # 可视化并保存 cross-attention map (1000)
    plot_and_save(
        cross_attention_map_numpy_1000,
        f"{filename_prefix}_cross_attn_1000.png",
        "Cross-Attention Map (1000)",
        crop=crop,
    )

    # 可视化并保存 self-attention map (1000)
    plot_and_save(
        self_attention_map_numpy_1000,
        f"{filename_prefix}_self_attn_1000.png",
        "Self-Attention Map (1000)",
        crop=crop,
    )



def fn_show_attention(
    cross_attention_maps,
    self_attention_maps,
    indices=None,
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
    # indices = [index - 1 for index in indices]
    if indices is None:
        indices = range(cross_attention_maps.size(2))

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
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / len(self_attention_map_per_token_list)
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
            (self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        
        self_attention_map_list.append(norm_self_attention_map_per_token)

    # tensor to numpy
    cross_attention_map_numpy       = torch.stack(cross_attention_map_list, dim=0).cpu().detach().numpy()
    self_attention_map_numpy        = torch.stack(self_attention_map_list, dim=0).cpu().detach().numpy()

    return cross_attention_map_numpy, self_attention_map_numpy


import cv2

def fn_get_otsu_mask(x):

    x_numpy = x
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