"""Endpoint crop-and-resize compatible with DenseSfM's RoIAlign extension.

This is NOT torchvision's half-pixel ROIAlign. Out-of-image sampling positions
are entirely zero, matching the upstream crop_and_resize CUDA implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RoIAlign(nn.Module):
    def __init__(self, crop_height, crop_width, transform_fpcoor=False):
        super().__init__()
        if transform_fpcoor:
            raise NotImplementedError('Only the endpoint convention is supported')
        self.height, self.width = crop_height, crop_width

    def forward(self, images, boxes, box_indices):
        h, w = images.shape[-2:]
        boxes = boxes.to(device=images.device, dtype=images.dtype)
        indices = box_indices.reshape(-1).long()
        if len(boxes) == 0:
            return images.new_empty((0, images.shape[1], self.height, self.width))
        xs = torch.linspace(0, 1, self.width, device=images.device, dtype=images.dtype)
        ys = torch.linspace(0, 1, self.height, device=images.device, dtype=images.dtype)
        x = boxes[:, 0, None] + (boxes[:, 2]-boxes[:, 0])[:, None]*xs
        y = boxes[:, 1, None] + (boxes[:, 3]-boxes[:, 1])[:, None]*ys
        xx = x[:, None, :].expand(-1, self.height, -1)
        yy = y[:, :, None].expand(-1, -1, self.width)
        valid = (xx >= 0) & (xx <= w-1) & (yy >= 0) & (yy <= h-1)
        grid = torch.stack((2*xx/(w-1)-1, 2*yy/(h-1)-1), dim=-1)
        result = images.new_empty((len(boxes), images.shape[1], self.height, self.width))
        # Group grids so each full image is sampled once, without expanding it
        # to one copy per keypoint (which otherwise exhausts GPU memory).
        for index in indices.unique():
            mask = indices == index
            g = grid[mask].reshape(1, -1, self.width, 2)
            sampled = F.grid_sample(images[index:index+1], g, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)
            result[mask] = sampled.reshape(images.shape[1], -1, self.height, self.width).permute(1, 0, 2, 3)
        return result * valid[:, None]
