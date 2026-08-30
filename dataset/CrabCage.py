import cv2
import torch
from torch.utils.data import Dataset, get_worker_info
from torchvision.transforms import Compose
from glob import glob
import os
import random
import numpy as np
from util.utils import fill_depth_map
import matplotlib.pyplot as plt

from dataset.utils.transform import Resize, NormalizeImage, PrepareForNet, Crop
from util.detect_period import detect_period


class CrabCage(Dataset):
    def __init__(self, mode, size=(518, 518), detect_target=True, detect_period=True):
        
        self.mode = mode
        self.size = size
        self.max_depth = 600
        self.detect_target = detect_target
        self.detect_period = detect_period
        self.fx = 1380.322
        self.B = 119.799

        self.fx_rectify_factor = 1.061
        self.fx = self.fx * self.fx_rectify_factor

        root = r'F:\dataset\stereo_matching_dataset'
        self.left_image_paths = sorted(glob(os.path.join(root, mode, '*/color_left_*.jpg')))
        self.right_image_paths = sorted(glob(os.path.join(root, mode, '*/color_right_*.jpg')))
        self.left_normal_paths = sorted(glob(os.path.join(root, mode, '*/normal_left_*.png')))
        self.right_normal_paths = sorted(glob(os.path.join(root, mode, '*/normal_right_*.png')))
        self.target_mask_left_paths = sorted(glob(os.path.join(root, mode, '*/target_mask_left_*.png')))
        self.target_mask_right_paths = sorted(glob(os.path.join(root, mode, '*/target_mask_right_*.png')))
        assert os.path.exists(self.left_normal_paths[0]), 'Normal map should be available before training.'

        if 'val' in mode:
            self.depth_paths = sorted(glob(os.path.join(root, mode, '*/depth_*.png')))
        
        assert len(self.left_image_paths) == len(self.right_image_paths), 'Left and right image counts do not match.'
        if 'val' in mode:
            assert len(self.left_image_paths) == len(self.depth_paths), 'Image and depth counts do not match.'
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if 'train' in self.mode else []))
        
        if "train" in self.mode:
            print(f'Load {len(self.left_image_paths)} training samples.')
        else:
            print(f'Load {len(self.left_image_paths)} validation samples.')

    
    def __getitem__(self, item):

        left_img_path = self.left_image_paths[item]
        right_img_path = self.right_image_paths[item]
        depth_path = self.depth_paths[item] if 'val' in self.mode else None

        left_image = cv2.imread(left_img_path)
        left_image_raw = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) / 255.0

        right_image = cv2.imread(right_img_path)
        right_image_raw = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB) / 255.0

        left_normal = cv2.imread(self.left_normal_paths[item], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 255.0 * 2 - 1
        right_normal = cv2.imread(self.right_normal_paths[item], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 255.0 * 2 - 1
        target_mask_left = cv2.imread(self.target_mask_left_paths[item], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 255.0 > 0.5
        target_mask_right = cv2.imread(self.target_mask_right_paths[item], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 255.0 > 0.5


        H, W = left_image.shape[:2]
              
        if 'val' in self.mode:
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) * 0.1  # 单位为mm
            depth[depth <= 0] = np.nan

            depth = fill_depth_map(depth)
            disp = self.fx * self.B / depth

        else:
            depth = np.zeros_like(left_image[:, :, 0]) - 1
            disp = np.zeros_like(left_image[:, :, 0]) - 1
        
        mask = ~np.isnan(depth)
        
        sample = self.transform({
            'left_image': left_image, 
            'left_image_raw': left_image_raw,
            'right_image': right_image,
            'right_image_raw': right_image_raw,
            'depth': depth,
            'disp': disp,
            'target_mask_left': target_mask_left,
            'target_mask_right': target_mask_right,
            'left_normal': left_normal,
            'right_normal': right_normal,
            'mask': mask,
        })

        sample['left_image'] = torch.from_numpy(sample['left_image'])
        sample['right_image'] = torch.from_numpy(sample['right_image'])
        sample['left_image_raw'] = torch.from_numpy(sample['left_image_raw'].astype(np.uint8)).permute(1, 2, 0)
        sample['right_image_raw'] = torch.from_numpy(sample['right_image_raw'].astype(np.uint8)).permute(1, 2, 0)
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['disp'] = torch.from_numpy(sample['disp'])
        sample['target_mask_left'] = torch.from_numpy(sample['target_mask_left']).bool()
        sample['target_mask_right'] = torch.from_numpy(sample['target_mask_right']).bool()
        sample['mask'] = torch.from_numpy(sample['mask'])
        sample['left_normal'] = torch.from_numpy(sample['left_normal'])
        sample['right_normal'] = torch.from_numpy(sample['right_normal'])

        period_left_list = detect_period(sample['left_image_raw'].numpy())
        period_right_list = detect_period(sample['right_image_raw'].numpy())
        min_diff = float('inf')
        best_left, best_right = 0.0, 0.0
        
        if self.detect_period:
            for p_l in period_left_list:
                for p_r in period_right_list:
                    diff = abs(p_l - p_r)
                    if diff < min_diff:
                        min_diff = diff
                        best_left = p_l
                        best_right = p_r
            
            period = (best_left + best_right) / 2.0
        
        else: period = 0.0

        sample['period'] = period
        sample['fx'] = self.fx
        sample['B'] = self.B

        sample['left_image_path'] = self.left_image_paths[item]
        sample['size'] = (H, W)
        sample['scale_factor'] = float(sample['scale_factor'])

        return sample

    def __len__(self):
        return len(self.left_image_paths)