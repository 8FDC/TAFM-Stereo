import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from glob import glob
import os
import random
import numpy as np
from util.utils import fill_depth_map
from .utils.utils import read_pfm
from typing import Literal

from dataset.utils.transform import Resize, NormalizeImage, PrepareForNet, Crop


class UWStereo(Dataset):
    def __init__(self, mode, size=(518, 518), scene:Literal['coral', 'default', 'industry', 'ship']='coral'):
        
        self.mode = mode
        self.size = size
        self.fx = 1050.0
        self.B = 1.0

        # root = fr'F:\dataset\sceneflow\flyingthings3d'

        self.left_image_paths = sorted(glob(rf'F:\dataset\uwstereo\UWScene\{scene}\images\left\*.png'))
        self.right_image_paths = sorted(glob(rf'F:\dataset\uwstereo\UWScene\{scene}\images\right\*.png'))
        self.disp_paths = sorted(glob(rf'F:\dataset\uwstereo\UWScene\{scene}\disparity\*.pfm'))
        self.left_normal_paths = sorted(glob(rf'F:\dataset\uwstereo\UWScene\{scene}\normals\left\*.png'))
        assert len(self.left_image_paths) == len(self.right_image_paths) == len(self.disp_paths)

        random_idx = list(range(len(self.left_image_paths)))
        random.seed(128)
        random.shuffle(random_idx)

        if self.mode == 'train':
            random_idx = random_idx[:int(len(random_idx)*0.8)]
        else:
            random_idx = list(set(random_idx) - set(random_idx[:int(len(random_idx)*0.8)]))
            
        self.left_image_paths = [self.left_image_paths[i] for i in random_idx]
        self.right_image_paths = [self.right_image_paths[i] for i in random_idx]
        self.disp_paths = [self.disp_paths[i] for i in random_idx]
        self.left_normal_paths = [self.left_normal_paths[i] for i in random_idx]



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
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
        self.transform_wo_normalize = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))

        if self.mode == 'train':
            print(f'Load {len(self.left_image_paths)} training samples.')
        else:
            print(f'Load {len(self.left_image_paths)} validation samples.')

    
    def __getitem__(self, item):

        left_img_path = self.left_image_paths[item]
        right_img_path = self.right_image_paths[item]
        disp_path = self.disp_paths[item] if self.mode == 'val' else None

        left_image = cv2.imread(left_img_path)
        left_image_raw = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) / 255.0

        right_image = cv2.imread(right_img_path)
        right_image_raw = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB) / 255.0

        left_normal = cv2.imread(self.left_normal_paths[item], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 255.0 * 2 - 1
        target_mask_left = np.zeros_like(left_image[:, :, 0], dtype=bool)
        target_mask_right = np.zeros_like(left_image[:, :, 0], dtype=bool)


        H, W = left_image.shape[:2]
              
        if self.mode == 'val':
            disp = read_pfm(disp_path)
            if disp.ndim == 3:
                disp = disp[..., 0]

            disp = disp.astype(np.float32)
            disp[disp <= 0] = np.nan

            depth = self.fx * self.B / disp

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
        
        period = 0.0

        sample['period'] = period
        sample['fx'] = self.fx
        sample['B'] = self.B

        sample['left_image_path'] = self.left_image_paths[item]
        sample['size'] = (H, W)
        sample['scale_factor'] = float(sample['scale_factor'])

        return sample

    def __len__(self):
        return len(self.left_image_paths)