import re

import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import L1Loss, MSELoss
from math import e

from .ssim import SSIM
from util.warping import warp_backward as warp
from util.depth_transfer import Depth_transfer
from .utils import convert_K, as_disp_hw, resample_feat_to_disp, census_transform, gradient_x, gradient_y
from util.depth_to_normal import Depth2Normal

class Loss(nn.Module):  
    def __init__(
            self,
            weighted=False,
            ordinal_weight=0.0,
            ordinal_patch_radius=3,
            ordinal_pair_stride=1,
            ordinal_rel_depth_delta=0.02,
            ordinal_disp_margin=0.1,
            ):
        super().__init__()

        self.weighted = weighted
        self.ordinal_weight = float(ordinal_weight)
        self.ordinal_patch_radius = int(ordinal_patch_radius)
        self.ordinal_pair_stride = max(1, int(ordinal_pair_stride))
        self.ordinal_rel_depth_delta = float(ordinal_rel_depth_delta)
        self.ordinal_disp_margin = float(ordinal_disp_margin)

        """图像重建相似性损失(SSIM)"""
        self.ssim_loss_fn = SSIM()

        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        
        self.depth_2_normal = Depth2Normal()
        


    """像素级光度损失(基于重建图像)"""    
    def photometric_loss(self, left_image, left_image_recon, mask, weight=None):

        if weight is None:
            weight = torch.ones_like(left_image[:, :, :, :])
        else:
            weight = weight.unsqueeze(1).repeat(1, 3, 1, 1)

        loss_map = torch.abs(left_image - left_image_recon) * weight
        valid = mask.repeat(1, 3, 1, 1)
        
        loss = torch.sum(loss_map[valid]) / torch.sum(valid)

        return loss


    def ssim_loss(self, left_image, left_image_recon, mask, weight=None):

        if weight is not None:
            weight = weight.unsqueeze(1).repeat(1, 3, 1, 1)
        else:
            weight = torch.ones_like(left_image)

        ssim_map = self.ssim_loss_fn(left_image, left_image_recon, mask) 
        ssim_dist = (1 - ssim_map) / 2.0
        
        loss = torch.nansum(ssim_dist * weight) / torch.sum(mask)  # normalize by number of valid pixels
        
        return loss
    

    """梯度平滑性损失"""
    # 加入边缘信息
    def gradient_smoothness(self, disp, K=3):
        
        smoothness_x = 0.0
        smoothness_y = 0.0

        for k in range(1, K+1):
            disp_dx = torch.abs(disp[:, :, :-k] - disp[:, :, k:])
            disp_dy = torch.abs(disp[:, :-k, :] - disp[:, k:, :])

            smoothness_x += disp_dx.mean()
            smoothness_y += disp_dy.mean()

        loss = 1 / K * (smoothness_x + smoothness_y)

        return loss


    """边缘平滑性损失(基于图像边缘)"""
    def edge_aware_smoothness(self, disp, image):
        disp_dx = torch.abs(disp[:, :, :-1] - disp[:, :, 1:])
        disp_dy = torch.abs(disp[:, :-1, :] - disp[:, 1:, :])

        image_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        image_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)

        weights_x = torch.exp(-image_dx)
        weights_y = torch.exp(-image_dy)

        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y

        return smoothness_x.mean() + smoothness_y.mean()


    """左右深度一致性损失"""
    def lr_depth_consistency(self, depth_L, depth_R, disp_pred_L):
        depth_R = depth_R.unsqueeze(1)
        depth_L = depth_L.unsqueeze(1)
        depth_L_recon, mask = warp(depth_R, disp_pred_L)
        loss = torch.mean(torch.abs(depth_L - depth_L_recon)[mask])
        
        return loss
    
    

    """左右特征一致性损失"""
    def lr_feature_consistency(self, feat_L, feat_R, disp_L, weights=None):

        loss = 0.0

        for wi, fL, fR in zip(weights, feat_L, feat_R):
            fL = resample_feat_to_disp(fL, disp_L)
            fR = resample_feat_to_disp(fR, disp_L)
            fR_warped, mask_L = warp(fR, disp_L.unsqueeze(1))

            fL_n  = F.normalize(fL, dim=1, eps=1e-6)
            fRw_n = F.normalize(fR_warped, dim=1, eps=1e-6)

            mask_L = mask_L.float()
            cos_L = (fL_n * fRw_n).sum(dim=1, keepdim=True)
            loss_L = ((1.0 - cos_L) * mask_L).sum() / mask_L.sum().clamp_min(1.0)
            loss += float(wi) * loss_L

        return loss
    


    """单目深度MAE损失"""
    def mono_depth_loss(self, depth_pred, disp, fx, B, valid_mask):

        disp_2_depth = fx.unsqueeze(-1).unsqueeze(-1) * B.unsqueeze(-1).unsqueeze(-1) / (disp + 1e-6)
        loss = self.MAE(depth_pred[valid_mask], disp_2_depth[valid_mask])

        return loss



    """代理损失"""
    def proxy_loss(self, disp_pred, disp_proxy):

        disp_proxy = disp_proxy * disp_proxy.shape[-1]
        
        loss = self.MSE(disp_pred, disp_proxy) / disp_proxy.shape[-1]**2
        
        return loss



    """Census变换损失"""
    def census_loss(self, img1, img2):
        c1 = census_transform(img1)
        c2 = census_transform(img2)

        diff = (c1 - c2)**2
        diff = diff / (0.1 + diff)

        return diff.mean()



    """图像梯度一致性损失"""
    def gradient_consistency_loss(self, img, img_recon, mask=None, alpha=10.0):
        grad_x1 = gradient_x(img)
        grad_x2 = gradient_x(img_recon)
        grad_y1 = gradient_y(img)
        grad_y2 = gradient_y(img_recon)
        weight_x = torch.exp(-alpha * torch.abs(grad_x1))
        weight_y = torch.exp(-alpha * torch.abs(grad_y1))

        loss_x = torch.abs(grad_x1 - grad_x2) * weight_x
        loss_y = torch.abs(grad_y1 - grad_y2) * weight_y

        if mask is not None:
            # normalize mask to (B,1,H,W)
            if mask.dim() == 3:
                mask_ = mask.unsqueeze(1)
            else:
                mask_ = mask
            mask_ = mask_.float().to(loss_x.dtype)

            # expand mask to match channel dim if needed
            if mask_.shape[1] != loss_x.shape[1]:
                mask_ = mask_.expand(-1, loss_x.shape[1], -1, -1)

            # mask spatial size must match loss_x/loss_y sizes:
            # loss_x has width reduced by 1 (W-1), same height
            # loss_y has height reduced by 1 (H-1), same width
            # create cropped masks for x and y respectively
            mask_x = mask_
            mask_y = mask_
            if mask_x.shape[-1] != loss_x.shape[-1]:
                mask_x = mask_x[..., :loss_x.shape[-1]]
            if mask_y.shape[-2] != loss_y.shape[-2]:
                mask_y = mask_y[..., :loss_y.shape[-2], :]

            # compute masked sums and normalize by combined valid count
            masked_sum = (loss_x * mask_x).sum() + (loss_y * mask_y).sum()
            denom = (mask_x.sum() + mask_y.sum()).clamp_min(1.0)
            loss = masked_sum / denom
        else:
            loss = loss_x.mean() + loss_y.mean()

        return loss
    

    """尺度不变对数损失"""
    def si_log_loss(self, depth_pred, depth_gt, mask):
        log_diff = torch.log(depth_pred[mask] + 1e-6) - torch.log(depth_gt[mask] + 1e-6)
        if log_diff.numel() == 0:
            return torch.zeros((), device=depth_pred.device, dtype=depth_pred.dtype)

        variance = (log_diff ** 2).mean() - (log_diff.mean() ** 2)
        loss = torch.sqrt(torch.clamp_min(variance, 0.0))
        
        return loss

    def normal_consistency_loss(self, fx, depth1, depth2):
        
        intrinsics = torch.zeros((depth1.shape[0], 3, 3), device=depth1.device)
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fx
        intrinsics[:, 2, 2] = 1.0
        intrinsics[:, 2, 0] = depth1.shape[-1] / 2
        intrinsics[:, 2, 1] = depth1.shape[-2] / 2
        
        if len(depth1.shape) == 3: depth1 = depth1.unsqueeze(1)
        if len(depth2.shape) == 3: depth2 = depth2.unsqueeze(1)

        mask1 = torch.ones_like(depth1).bool()
        mask2 = torch.ones_like(depth2).bool()
        
        normal1, normal_mask1 = self.depth_2_normal(
            depth1,
            intrinsics,
            mask1,
            1.0
        )
        
        normal2, normal_mask2 = self.depth_2_normal(
            depth2,
            intrinsics,
            mask2,
            1.0
        )
        
        cos_sim = F.cosine_similarity(normal1, normal2, dim=1)
        valid_mask = normal_mask1.squeeze(1) & normal_mask2.squeeze(1)

        loss = 1.0 - cos_sim[valid_mask].mean()
        return loss

    def forward(self, sample, output):

        fx = sample['fx']
        B = sample['B']
        scale_factor = sample['scale_factor'].view(-1, 1, 1)


        left_image = sample["left_image_raw"].permute(0, 3, 1, 2) / 255
        right_image = sample["right_image_raw"].permute(0, 3, 1, 2) / 255
        # left_image = sample["left_image"]
        # right_image = sample["right_image"]

        # gt_mask = sample['mask'].bool()

        weight_map_left = torch.ones_like(left_image[:, :1, :, :]).squeeze(1)
        weight_map_right = torch.ones_like(right_image[:, :1, :, :]).squeeze(1)
        if 'target_mask_left' in sample and self.weighted:
            weight_map_left[sample['target_mask_left']] = weight_map_left[sample['target_mask_left']] * 3
        if 'target_mask_right' in sample and self.weighted:
            weight_map_right[sample['target_mask_right']] = weight_map_right[sample['target_mask_right']] * 3


        depth_L = output['depth_prediction_left']
        depth_R = output['depth_prediction_right']
        cost_volume_ordinal = output.get(
            'cost_volume_ordinal_loss',
            torch.tensor(0.0).to(device=left_image.device),
        )
        disp_list_left = output['disp_predictions_left']
        disp_list_right = output['disp_predictions_right']
        # mono_features_left = output['mono_features_left']
        # mono_features_right = output['mono_features_right']

        disp_L = disp_list_left[-1].clamp(min=1e-6)
        disp_R = disp_list_right[-1].clamp(min=1e-6)
        # disp_L_up = disp_L / scale_factor
        # disp_R_up = disp_R / scale_factor
        


        """Stereo Loss (left)"""
        image_recon_l, recon_mask_l = warp(right_image, disp_L, invert=True)
        image_recon_r, recon_mask_r = warp(left_image, disp_R, invert=False)

        ssim_stereo_l = torch.tensor(0.0).to(device=left_image.device)
        photometric_stereo_l = torch.tensor(0.0).to(device=left_image.device)
        edge_stereo_l = torch.tensor(0.0).to(device=left_image.device)
        census_stereo_l = torch.tensor(0.0).to(device=left_image.device)
        gradient_stereo_l = torch.tensor(0.0).to(device=left_image.device)
        # gradient_smoothness_stereo_l = torch.tensor(0.0).to(device=left_image.device)
        lr_disp_consis_l = torch.tensor(0.0).to(device=left_image.device)

        ssim_stereo_l = 0.50 * self.ssim_loss(left_image, image_recon_l, recon_mask_l, weight=weight_map_left)
        photometric_stereo_l = 6.00 * self.photometric_loss(left_image, image_recon_l, recon_mask_l, weight_map_left)
        edge_stereo_l = 0.1 * self.edge_aware_smoothness(disp_L, left_image)
        # census_stereo_l = self.census_loss(left_image, left_image_recon)
        gradient_stereo_l = 5 * self.gradient_consistency_loss(left_image, image_recon_l, recon_mask_l)
        # gradient_smoothness_stereo_l = 0.05 * self.gradient_smoothness(disp_L)
        lr_disp_consis_l = 0.2*self.MAE(
            disp_L.clamp(min=1e-6)[recon_mask_l.squeeze(1)], 
            warp(disp_R.unsqueeze(1), disp_L, invert=True)[0].squeeze(1).detach()[recon_mask_l.squeeze(1)]
            )
        stereo_loss_left =\
            ssim_stereo_l +\
            photometric_stereo_l +\
            census_stereo_l +\
            gradient_stereo_l +\
            edge_stereo_l +\
            lr_disp_consis_l

        

        """Stereo Loss (right)"""
        ssim_stereo_r = torch.tensor(0.0).to(device=right_image.device)
        photometric_stereo_r = torch.tensor(0.0).to(device=right_image.device)
        edge_stereo_r = torch.tensor(0.0).to(device=right_image.device)
        census_stereo_r = torch.tensor(0.0).to(device=right_image.device)
        gradient_stereo_r = torch.tensor(0.0).to(device=right_image.device)
        # gradient_smoothness_stereo_r = torch.tensor(0.0).to(device=right_image.device)
        lr_disp_consis_r = torch.tensor(0.0).to(device=right_image.device)

        ssim_stereo_r = 0.50 * self.ssim_loss(right_image, image_recon_r, recon_mask_r, weight=weight_map_right)
        photometric_stereo_r = 6.00 * self.photometric_loss(right_image, image_recon_r, recon_mask_r, weight_map_right)
        edge_stereo_r = 0.1 * self.edge_aware_smoothness(disp_R, right_image)
        # census_stereo_r = self.census_loss(right_image, right_image_recon)
        gradient_stereo_r = 5*self.gradient_consistency_loss(right_image, image_recon_r, recon_mask_r)
        # gradient_smoothness_stereo_r = 0.05 * self.gradient_smoothness(disp_R)
        lr_disp_consis_r = 0.2*self.MAE(
            disp_R.clamp(min=1e-6)[recon_mask_r.squeeze(1)], 
            warp(disp_L.unsqueeze(1), disp_R, invert=False)[0].squeeze(1).detach()[recon_mask_r.squeeze(1)]
            )
        stereo_loss_right =\
            ssim_stereo_r +\
            photometric_stereo_r +\
            census_stereo_r +\
            gradient_stereo_r +\
            edge_stereo_r +\
            lr_disp_consis_r



        """Mono loss (Left)"""
        disp_L_ = fx.unsqueeze(-1).unsqueeze(-1) * B.unsqueeze(-1).unsqueeze(-1) / (depth_L + 1e-6) * scale_factor
        image_recon_l_, recon_mask_l_ = warp(right_image, disp_L_, invert=True)
        depth_L_ = fx.unsqueeze(-1).unsqueeze(-1) * B.unsqueeze(-1).unsqueeze(-1) / (disp_L + 1e-6) * scale_factor

        ssim_mono_l = torch.tensor(0.0).to(device=left_image.device)
        edge_mono_l = torch.tensor(0.0).to(device=left_image.device)
        photometric_mono_l = torch.tensor(0.0).to(device=left_image.device)
        census_mono_l = torch.tensor(0.0).to(device=left_image.device)
        gradient_mono_l = torch.tensor(0.0).to(device=left_image.device)
        # gradient_smoothness_mono_l = torch.tensor(0.0).to(device=left_image.device)
        si_log_mono_l = torch.tensor(0.0).to(device=left_image.device)
        normal_consistency_mono_l = torch.tensor(0.0).to(device=left_image.device)

        
        ssim_mono_l = 0.50 * self.ssim_loss(left_image, image_recon_l_, recon_mask_l_, weight=weight_map_left)
        photometric_mono_l = 6.00 * self.photometric_loss(left_image, image_recon_l_, recon_mask_l_, weight=weight_map_left)
        edge_mono_l = 0.1 * self.edge_aware_smoothness(disp_L_, left_image)
        gradient_mono_l = 5*self.gradient_consistency_loss(left_image, image_recon_l_, recon_mask_l_)
        # gradient_smoothness_mono_l = 0.05 * self.gradient_smoothness(disp_L_)
        si_log_mono_l = 5 * self.si_log_loss(disp_L.clamp(min=1e-6), disp_L_, recon_mask_l_.squeeze(1))
        normal_consistency_mono_l = 0.5 * self.normal_consistency_loss(fx, depth_L, depth_L_)

        mono_loss_l =\
            ssim_mono_l +\
            photometric_mono_l +\
            gradient_mono_l +\
            edge_mono_l +\
            si_log_mono_l +\
            normal_consistency_mono_l


        """Mono loss (Right)"""
        disp_R_ = fx.unsqueeze(-1).unsqueeze(-1) * B.unsqueeze(-1).unsqueeze(-1) / (depth_R + 1e-6) * scale_factor
        right_image_recon_d, recon_mask_r_d = warp(left_image, disp_R_, invert=False)
        depth_R_ = fx.unsqueeze(-1).unsqueeze(-1) * B.unsqueeze(-1).unsqueeze(-1) / (disp_R + 1e-6) * scale_factor
        
        ssim_mono_r = torch.tensor(0.0).to(device=right_image.device)
        edge_mono_r = torch.tensor(0.0).to(device=right_image.device)
        photometric_mono_r = torch.tensor(0.0).to(device=right_image.device)
        census_mono_r = torch.tensor(0.0).to(device=right_image.device)
        gradient_mono_r = torch.tensor(0.0).to(device=right_image.device)
        # gradient_smoothness_mono_r = torch.tensor(0.0).to(device=right_image.device)
        normal_consistency_mono_r = torch.tensor(0.0).to(device=right_image.device)
        si_log_mono_r = torch.tensor(0.0).to(device=right_image.device)

        ssim_mono_r = 0.50 * self.ssim_loss(right_image, right_image_recon_d, recon_mask_r_d, weight=weight_map_right)
        photometric_mono_r = 6.00 * self.photometric_loss(right_image, right_image_recon_d, recon_mask_r_d, weight=weight_map_right)
        edge_mono_r = 0.1 * self.edge_aware_smoothness(disp_R_, right_image)
        gradient_mono_r = 5*self.gradient_consistency_loss(right_image, right_image_recon_d, mask=recon_mask_r_d)
        # gradient_smoothness_mono_r = 0.05 * self.gradient_smoothness(disp_R_)
        si_log_mono_r = 5*self.si_log_loss(disp_R.clamp(min=1e-6), disp_R_, recon_mask_r_d.squeeze(1))
        normal_consistency_mono_r = 0.5 * self.normal_consistency_loss(fx, depth_R, depth_R_)

        mono_loss_r =\
            ssim_mono_r +\
            photometric_mono_r +\
            gradient_mono_r +\
            edge_mono_r +\
            si_log_mono_r +\
            normal_consistency_mono_r


        ordinal_stereo = self.ordinal_weight * cost_volume_ordinal
        total_loss = stereo_loss_left + stereo_loss_right + mono_loss_l + mono_loss_r + normal_consistency_mono_l + normal_consistency_mono_r + ordinal_stereo

        
        return (
            total_loss,
            {
                "stereo_loss":{
                    "total": stereo_loss_left + stereo_loss_right,
                    "ssim": ssim_stereo_l + ssim_stereo_r,
                    "photometric": photometric_stereo_l + photometric_stereo_r,
                    "census": census_stereo_l + census_stereo_r,
                    "edge": edge_stereo_l + edge_stereo_r,
                    "gradient": gradient_stereo_l + gradient_stereo_r,
                    "lr_disp_consistency": lr_disp_consis_l + lr_disp_consis_r,
                    "cv_ordinal_depth": ordinal_stereo
                },
                "mono_loss":{
                    "total": mono_loss_l + mono_loss_r,
                    "ssim": ssim_mono_l + ssim_mono_r,
                    "photometric": photometric_mono_l + photometric_mono_r,
                    "census": census_mono_l + census_mono_r,
                    "edge": edge_mono_l + edge_mono_r,
                    "gradient": gradient_mono_l + gradient_mono_r,
                    "disp_depth": si_log_mono_l + si_log_mono_r,
                    "normal": normal_consistency_mono_l + normal_consistency_mono_r
                }
            }
            )
