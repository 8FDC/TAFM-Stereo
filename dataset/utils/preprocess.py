import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from dataset.utils.transform import NormalizeImage, PrepareForNet, Resize
from util.detect_period import detect_period


def preprocess(
    left_img_path,
    right_img_path,
    left_normal_path,
    left_mask_path,
    right_mask_path,
    depth_path,
    size=(518, 518),
    fx=1380.322,
    baseline=119.799,
    fx_rectify_factor=1.061,
    detect_period_flag=True,
):
    mode = "stereo"
    if (right_img_path is None) or (not os.path.exists(right_img_path)):
        mode = "mono"

    net_w, net_h = size
    transform = Compose([
        Resize(
            width=net_w,
            height=net_h,
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    fx = fx * fx_rectify_factor

    # image
    left_bgr = cv2.imread(left_img_path)
    left_image_raw = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

    if mode == "stereo":
        right_bgr = cv2.imread(right_img_path)
        right_image_raw = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    else:
        right_image_raw = np.zeros_like(left_image_raw)

    left_image = left_image_raw / 255.0
    right_image = right_image_raw / 255.0

    # auxiliary maps
    left_normal = cv2.imread(left_normal_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 255.0 * 2 - 1
    target_mask_left = cv2.imread(left_mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 255.0 > 0.5

    if mode == "stereo" and right_mask_path is not None and os.path.exists(right_mask_path):
        target_mask_right = cv2.imread(right_mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 255.0 > 0.5
    else:
        target_mask_right = np.zeros_like(target_mask_left, dtype=bool)

    # depth / disp
    if depth_path is not None and os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) * 0.1
        depth[depth <= 0] = np.nan
        disp = fx * baseline / depth
    else:
        depth = np.zeros_like(left_image[:, :, 0], dtype=np.float32) - 1.0
        disp = np.zeros_like(left_image[:, :, 0], dtype=np.float32) - 1.0

    mask = ~np.isnan(depth)
    h, w = left_image.shape[:2]

    sample = transform({
        "left_image": left_image,
        "left_image_raw": left_image_raw,
        "right_image": right_image,
        "right_image_raw": right_image_raw,
        "depth": depth,
        "disp": disp,
        "target_mask_left": target_mask_left,
        "target_mask_right": target_mask_right,
        "left_normal": left_normal,
        "mask": mask,
    })

    sample["left_image"] = torch.from_numpy(sample["left_image"]).unsqueeze(0).to("cuda")
    sample["right_image"] = torch.from_numpy(sample["right_image"]).unsqueeze(0).to("cuda")
    sample["left_image_raw"] = torch.from_numpy(sample["left_image_raw"].astype(np.uint8)).permute(1, 2, 0)
    sample["right_image_raw"] = torch.from_numpy(sample["right_image_raw"].astype(np.uint8)).permute(1, 2, 0)
    sample["depth"] = torch.from_numpy(sample["depth"])
    sample["disp"] = torch.from_numpy(sample["disp"])
    sample["target_mask_left"] = torch.from_numpy(sample["target_mask_left"]).bool().unsqueeze(0).to("cuda")
    sample["target_mask_right"] = torch.from_numpy(sample["target_mask_right"]).bool().unsqueeze(0).to("cuda")
    sample["mask"] = torch.from_numpy(sample["mask"]).unsqueeze(0).to("cuda")
    sample["left_normal"] = torch.from_numpy(sample["left_normal"]).unsqueeze(0).to("cuda")

    # period (aligned with CrabCage style + empty-list fallback)
    period = 0.0
    if detect_period_flag:
        period_left_list = detect_period(sample["left_image_raw"].numpy())
        period_right_list = detect_period(sample["right_image_raw"].numpy())
        if len(period_left_list) > 0 and len(period_right_list) > 0:
            min_diff = float("inf")
            best_left, best_right = 0.0, 0.0
            for p_l in period_left_list:
                for p_r in period_right_list:
                    diff = abs(p_l - p_r)
                    if diff < min_diff:
                        min_diff = diff
                        best_left, best_right = p_l, p_r
            period = (best_left + best_right) / 2.0

    sample["period"] = torch.tensor(period, dtype=torch.float32).unsqueeze(0).to("cuda")
    sample["fx"] = torch.tensor(fx, dtype=torch.float32).unsqueeze(0).to("cuda")
    sample["B"] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to("cuda")
    sample["left_image_path"] = left_img_path
    sample["right_image_path"] = right_img_path
    sample["size"] = (h, w)
    sample["scale_factor"] = torch.tensor(size[0] / min(h, w), dtype=torch.float32).unsqueeze(0).to("cuda")

    return sample