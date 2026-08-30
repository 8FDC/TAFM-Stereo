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

from dataset.utils.transform import Resize, NormalizeImage, PrepareForNet, Crop


class Monkaa(Dataset):
    def __init__(self, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        self.fx = 450.0
        self.B = 1.0

        root = r'F:/dataset/monkaa'
        # self.left_image_paths = sorted(glob(os.path.join(root, 'frames_cleanpass', '*\\left', '*.png')))
        # self.right_image_paths = sorted(glob(os.path.join(root, 'frames_cleanpass', '*\\right', '*.png')))
        # self.disp_paths = sorted(glob(os.path.join(root, 'disparity', '*\\left', '*.pfm')))

        self.left_image_paths = sorted(glob(os.path.join(root, 'frames_cleanpass', 'eating*\\left', '*.png')))
        self.right_image_paths = sorted(glob(os.path.join(root, 'frames_cleanpass', 'eating*\\right', '*.png')))
        self.disp_paths = sorted(glob(os.path.join(root, 'disparity', 'eating*\\left', '*.pfm')))

        assert len(self.left_image_paths) == len(self.right_image_paths) == len(self.disp_paths)

        random.seed(128)
        self.train_idx = random.sample(range(len(self.left_image_paths)), int(0.8 * len(self.left_image_paths)))
        self.val_idx = list(set(range(len(self.left_image_paths))) - set(self.train_idx))
        random.shuffle(self.train_idx)
        random.shuffle(self.val_idx)

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
            print(f'Load {len(self.train_idx)} training samples.')
        else:
            print(f'Load {len(self.val_idx)} validation samples.')

    
    def __getitem__(self, idx):

        item = self.train_idx[idx] if self.mode == 'train' else self.val_idx[idx]

        
        left_img_path = self.left_image_paths[item]
        right_img_path = self.right_image_paths[item]
        disp_path = self.disp_paths[item]

        left_image = cv2.imread(left_img_path)
        left_image_raw = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) / 255.0

        right_image = cv2.imread(right_img_path)
        right_image_raw = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB) / 255.0

        H, W = left_image.shape[:2]
        
        disp = read_pfm(disp_path)
        if disp.ndim == 3:
            disp = disp[..., 0]

        disp = disp.astype(np.float32)
        disp[disp <= 0] = np.nan

        depth = self.fx * self.B / disp

        mask = ~np.isnan(depth)
        
        sample = self.transform({
            'left_image': left_image,
            'left_image_raw': left_image_raw,
            'right_image': right_image,
            'right_image_raw': right_image_raw,
            'disp': disp,
            'depth': depth,
            'mask': mask,
        })

        sample['left_image'] = torch.from_numpy(sample['left_image'])
        sample['right_image'] = torch.from_numpy(sample['right_image'])
        sample['disp'] = torch.from_numpy(sample['disp'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['mask'] = torch.from_numpy(sample['mask'])

        sample['left_image_path'] = self.left_image_paths[item]
        sample['size'] = (H, W)

        return sample

    def __len__(self):
        return len(self.train_idx) if self.mode == 'train' else len(self.val_idx)