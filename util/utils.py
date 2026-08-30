from operator import is_
import os
import re
import numpy as np
import logging
from datetime import datetime
import shutil
import torch
import torch.nn.functional as F

logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger



def fill_depth_map(depth, method='linear'):

    from scipy import interpolate

    depth_copy = depth.copy().astype(np.float32)
    
    mask = np.isnan(depth_copy)
    
    if not np.any(mask): return depth_copy

    y, x = np.indices(depth_copy.shape)

    x_valid = x[~mask]
    y_valid = y[~mask]
    z_valid = depth_copy[~mask]

    depth_filled = interpolate.griddata(
        points=(x_valid, y_valid),
        values=z_valid,
        xi=(x, y),
        method=method
    )
    

    return depth_filled


def backup_workspace(timestamp=None, backup_dir=None):
    src_dir = './'
    
    if backup_dir is None:
        backup_dir = os.path.join(src_dir, 'backup')
    else:
        backup_dir = os.path.join(src_dir, backup_dir)

    if timestamp is not None:
        target_dir = os.path.join(backup_dir, f'backup_{timestamp}')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    target_dir = os.path.join(backup_dir, f'backup_{timestamp}')
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    exclude = {'outputs', 'backup', '__pycache__', ".git"}
    for item in os.listdir(src_dir):
        if item in exclude:
            continue
        item_path = os.path.join(src_dir, item)
        dest_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            shutil.copytree(item_path, dest_path, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(item_path, dest_path)


def rectify_disp_edge(disp, edge_width=144):

    is_tensor = False
    device = disp.device if isinstance(disp, torch.Tensor) else None

    if isinstance(disp, torch.Tensor):
        disp = disp.cpu().numpy()
        is_tensor = True

    disp[:edge_width, :] = np.nan
    disp[-edge_width:, :] = np.nan
    disp[:, :edge_width] = np.nan
    disp[:, -edge_width:] = np.nan

    # disp = fill_depth_map(disp, method='linear').astype(np.float32)
    disp = fill_depth_map(disp, method='nearest').astype(np.float32)

    if is_tensor:
        disp = torch.from_numpy(disp).to(device)

    return disp


def gen_disparity_mask(base_disparities, invert=True, out_size=None, threshold=0.5):

    if base_disparities.dim() == 4:
        base_disparities = base_disparities.squeeze(1)  # [B,H,W]

    B, H_d, W_d = base_disparities.shape
    H, W = (H_d, W_d) if out_size is None else out_size

    y = torch.arange(H_d, device=base_disparities.device, dtype=base_disparities.dtype)
    x = torch.arange(W_d, device=base_disparities.device, dtype=base_disparities.dtype)
    y, x = torch.meshgrid(y, x, indexing="ij")
    y = y.unsqueeze(0).expand(B, -1, -1)
    x = x.unsqueeze(0).expand(B, -1, -1)

    x = x + (-base_disparities if invert else base_disparities)

    y = 2.0 * y / max(H - 1, 1) - 1.0
    x = 2.0 * x / max(W - 1, 1) - 1.0
    grid = torch.stack((x, y), dim=-1)  # [B,H_d,W_d,2]

    ones = torch.ones((B, 1, H, W), device=base_disparities.device, dtype=base_disparities.dtype)
    soft_mask = F.grid_sample(
        ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    mask = soft_mask > threshold

    return mask


def gen_x_grad_map(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    grad_x = depth[:, :, 1:] - depth[:, :, :-1]
    grad_x = F.pad(grad_x, (0, 1), mode='replicate')

    return grad_x