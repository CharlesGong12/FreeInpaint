from typing import Tuple, Union
import numpy as np
from PIL import Image

def resize(image: Image, size: Union[int, Tuple[int, int]], resample=Image.BICUBIC):
    if isinstance(size, int):
        w, h = image.size
        aspect_ratio = w / h
        size = (min(size, int(size * aspect_ratio)),
                min(size, int(size / aspect_ratio)))
    return image.resize(size, resample=resample)

def get_bbox_from_mask(mask, relax_ratio=0):
    """
    Get bbox from mask
    Args:
        mask: np.ndarray, mask
        relax_ratio: float, relax ratio
    Returns:
        bbox, [x, y, x+w, y+h]
    """
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    if relax_ratio > 0:
        w = x_max - x_min
        h = y_max - y_min
        x_min = max(0, int(x_min - relax_ratio * w))
        x_max = min(mask.shape[1], int(x_max + relax_ratio * w))
        y_min = max(0, int(y_min - relax_ratio * h))
        y_max = min(mask.shape[0], int(y_max + relax_ratio * h))
            
    return [x_min, y_min, x_max, y_max]