import torch
import torch.nn.functional as F
import numpy as np



def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def convert_K(K, orig_h, orig_w, target_k):
    """
    假设居中裁剪(实际上这样计算出的cx和cy是不准确的)
    """
    s = target_k / orig_h
    new_w = orig_w * s
    crop_x = (new_w - target_k) / 2

    K_new = K.copy()
    K_new[0][0] *= s  # fx
    K_new[1][1] *= s  # fy
    K_new[0][2] = K[0][2] * s - crop_x  # cx
    K_new[1][2] = K[1][2] * s           # cy

    return K_new

def as_disp_hw(disp: torch.Tensor):
    """disp: [B,H,W] or [B,1,H,W] -> (H,W)"""
    if disp.dim() == 4:
        disp = disp.squeeze(1)
    return disp.shape[-2], disp.shape[-1]

def resample_feat_to_disp(feat: torch.Tensor, disp: torch.Tensor):
    """feat: [B,C,h,w] -> [B,C,H,W] where (H,W) from disp"""
    H, W = as_disp_hw(disp)
    if feat.shape[-2:] == (H, W):
        return feat
    return F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

def normalize(arry):
    return (arry - arry.min()) / (arry.max() - arry.min() + 1e-8)


def census_transform(img, patch_size=7):
    B, C, H, W = img.shape
    pad = patch_size // 2

    img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')      # [B, C, H+2p, W+2p]
    patches = F.unfold(img_pad, kernel_size=patch_size)             # [B, C*k*k, H*W]
    patches = patches.view(B, C, patch_size * patch_size, H, W)     # [B, C, k*k, H, W]

    center = img.unsqueeze(2)                                        # [B, C, 1, H, W]
    census = torch.clamp(patches - center, -1, 1)

    return census