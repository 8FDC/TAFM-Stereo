import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from depth_anything_v2.DepthAnythingV2 import DepthAnythingV2
from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.dpt import DPTHead, ReadoutBlock, DPTHead_FiLM, ScratchHead, ScratchHeadTaskAware
from .feature_transfer import Feat_transfer, Feat_transfer_cnet
from .submodule import BasicConv_IN, GradModulator, MultiBasicEncoder, context_upsample, GradModulator
from raft_stereo.corr import CorrBlock1D as corr_block
from raft_stereo.utils.utils import hor_coords_grid
from raft_stereo.update_disp import DispBasicMultiUpdateBlock
from .roi_attn import ROICrossAttention
from util.init_disp import init_disparity_piecewise
from util.utils import gen_x_grad_map
from sam_features_extraction.sam2.sam2.build_sam import build_sam2
from sam_features_extraction.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from sam_features_extraction.utils import detect_target, get_kmeans_points

SAM_FEAT_HEIGHTS = (148, 74, 37, 19)

autocast = torch.cuda.amp.autocast


def get_sam_feat_sizes(image_h, image_w):
    ratio = image_w / float(image_h)
    sizes = []
    for h in SAM_FEAT_HEIGHTS:
        w = int(h * ratio)
        sizes.append((h, w))
    return sizes


class Model(nn.Module):
    def __init__(
            self, 
            cfg,
            encoder_name='vits', 
            features=64, 
            out_channels=[48, 96, 192, 384], 
            use_bn=False, 
            use_clstoken=False, 
            use_FiLM=True,
            max_depth=900.0
            ):
        
        super(Model, self).__init__()

        self.cfg = cfg

        self.slow_fast_gru = False

        self.use_FiLM = use_FiLM
        self.max_depth = max_depth

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }

        mono_model_configs = {
            'vits': {'encoder_name': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder_name': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder_name': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder_name': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.encoder_name = encoder_name
        dim_list_ = mono_model_configs[encoder_name]['features']
        dim_list = []
        dim_list.append(dim_list_)


        """单目编码器和解码器"""
        # self.encoder = DINOv2(model_name=encoder_name)
        self.share_encoder = SAM2ImagePredictor(build_sam2(
            "sam_features_extraction/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam_features_extraction/sam2/checkpoints/sam2.1_hiera_small.pt" 
        ))
        # self.readout_block = ReadoutBlock(self.encoder.embed_dim, out_channels=out_channels, use_clstoken=use_clstoken)
        self.mono_decoder =  DPTHead_FiLM(features, use_bn, out_channels=out_channels, film_channels=cfg['hidden_dims'][0] + cfg['hidden_dims'][1]) if use_FiLM \
            else DPTHead(features, use_bn, out_channels=out_channels)
        # self.scratch_head = ScratchHead(features, use_bn, out_channels=out_channels)

        # state_dict_dpt = torch.load(f'./checkpoints/depth_anything_v2_{encoder_name}.pth', map_location='cpu')
        # self.encoder.load_state_dict(state_dict_dpt, strict=False)  # 仅加载DINOv2参数


        """用于提取特征的单目深度估计模型(不训练)"""
        self.dfm = DepthAnythingV2(**mono_model_configs[encoder_name])


        """梯度调制器"""
        self.stereo_grad_modulator = GradModulator()

        """单目特征转换器(将来会被替换为任务注意力模块)"""
        # self.feat_transfer = Feat_transfer(dim_list)
        # self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=128)

        """任务注意力模块"""
        self.share_task_head = ScratchHeadTaskAware(features=features, out_channels=out_channels)

        self.stereo_conv = BasicConv_IN(48, 96, kernel_size=3, padding=1, stride=1)  # 96-->48, 128-->96
        self.stereo_desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)


        """上下文特征提取器"""
        context_dims = cfg['hidden_dims']
        self.stereo_context_ectractor = MultiBasicEncoder(
            in_channels=8,
            output_dim=[cfg['hidden_dims'], context_dims], 
            norm_fn=cfg['context_norm'],
            downsample=cfg['n_downsample']
            )

        # """ROI交叉注意力模块"""
        # if cfg['use_roi_attn']:
        #     self.roi_cross_attn = ROICrossAttention(
        #         in_channels=96,
        #         num_heads=4,
        #         max_tokens=256,
        #         roi_gain=1.0,
        #         query_downsample=2,
        #         min_roi_pixels=8,
        #         update_background=False,
        #     )

        self.stereo_update_block = DispBasicMultiUpdateBlock(self.cfg, hidden_dims=cfg['hidden_dims'])

        self.stereo_context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], cfg['hidden_dims'][i]*3, 3, padding=3//2) for i in range(self.cfg['n_gru_layers'])])

        self.share_sam_feat_proj = nn.ModuleList([
            nn.Conv2d(256, out_channels[i], kernel_size=1, padding=0, stride=1)
            for i in range(len(out_channels))
        ])


    def initialize_disp(self, img):
        """ Disparity is represented as difference between two horizontal coordinate grids disp = hor_coords1 - hor_coords0"""
        N, _, H, W = img.shape

        hor_coords0 = hor_coords_grid(N, H, W).to(img.device)
        hor_coords1 = hor_coords_grid(N, H, W).to(img.device)

        return hor_coords0, hor_coords1

    def normalize(self, arry):
        return (arry - arry.min()) / (arry.max() - arry.min() + 1e-8)


    def init_disp_from_rel_depth(self, rel_depth, coords):
        init_disp = F.interpolate(
            rel_depth.unsqueeze(1),
            size=coords.shape[-2:],
            mode='bilinear',
            align_corners=True,
        )
        scale = self.cfg.get('relative_depth_disp_scale', self.cfg.get('disp_init_var', 1.0))
        shift = self.cfg.get('relative_depth_disp_shift', 0.0)
        return init_disp * scale + shift
    
    
    def upsample_disp(self, disp, mask):
        """ Upsample disp field [H/8, W/8, 1] -> [H, W, 1] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.cfg['n_downsample']
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(factor * disp, [3,3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor*H, factor*W)

    def resize_ordinal_map(self, x, size, mode):
        if x is None:
            return None
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if mode == 'bilinear':
            x = F.interpolate(x.float(), size=size, mode=mode, align_corners=False)
        else:
            x = F.interpolate(x.float(), size=size, mode=mode)
        return x.squeeze(1)

    def corr_level0_logits(self, corr):
        radius = self.cfg['corr_radius']
        base_hyp = 2 * radius + 1
        if self.cfg.get('beat_repetitive_texture', False):
            start = base_hyp
            end = 2 * base_hyp
            return corr[:, start:end]
        return corr[:, :base_hyp]

    def cost_volume_ordinal_prior_loss(self, corr, disp_center, rel_depth):
        if not self.cfg.get('use_cost_volume_ordinal_prior', False):
            return corr.new_zeros(())
        if self.cfg.get('ordinal_loss_weight', 0.0) <= 0.0:
            return corr.new_zeros(())
        if rel_depth is None:
            return corr.new_zeros(())

        logits = self.corr_level0_logits(corr)
        B, K, H, W = logits.shape
        rel_depth = self.resize_ordinal_map(rel_depth.detach(), (H, W), mode='bilinear').to(logits.device)

        temp = self.cfg.get('cv_ordinal_temperature', 0.2)
        sharpness = self.cfg.get('cv_ordinal_order_sharpness', 10.0)
        margin = self.cfg.get('ordinal_disp_margin', 0.1)
        rel_delta_thr = self.cfg.get('ordinal_rel_depth_delta', 0.02)
        radius = self.cfg.get('ordinal_patch_radius', 3)
        stride = max(1, self.cfg.get('ordinal_pair_stride', 1))

        prob = torch.softmax(logits / temp, dim=1)
        offsets = torch.linspace(
            -self.cfg['corr_radius'],
            self.cfg['corr_radius'],
            K,
            device=logits.device,
            dtype=logits.dtype,
        ).view(1, K, 1, 1)
        hyp_disp = disp_center.detach() + offsets

        losses = []
        for dy in range(-radius, radius + 1, stride):
            for dx in range(-radius, radius + 1, stride):
                if dy == 0 and dx == 0:
                    continue
                if dy < 0 or (dy == 0 and dx < 0):
                    continue

                y0_a, y1_a = max(0, dy), H + min(0, dy)
                x0_a, x1_a = max(0, dx), W + min(0, dx)
                y0_b, y1_b = max(0, -dy), H - max(0, dy)
                x0_b, x1_b = max(0, -dx), W - max(0, dx)

                rel_a = rel_depth[:, y0_a:y1_a, x0_a:x1_a]
                rel_b = rel_depth[:, y0_b:y1_b, x0_b:x1_b]

                rel_delta = rel_a - rel_b
                sign = torch.sign(rel_delta)
                pair_mask = (torch.abs(rel_delta) > rel_delta_thr) & (sign != 0)
                if not torch.any(pair_mask):
                    continue

                prob_a = prob[:, :, y0_a:y1_a, x0_a:x1_a]
                prob_b = prob[:, :, y0_b:y1_b, x0_b:x1_b]
                disp_a = hyp_disp[:, :, y0_a:y1_a, x0_a:x1_a]
                disp_b = hyp_disp[:, :, y0_b:y1_b, x0_b:x1_b]

                order_score = sign.unsqueeze(1).unsqueeze(2) * (
                    disp_a.unsqueeze(2) - disp_b.unsqueeze(1)
                )
                order_prob = torch.sigmoid(sharpness * (order_score - margin))
                pair_prob = (prob_a.unsqueeze(2) * prob_b.unsqueeze(1) * order_prob).sum(dim=(1, 2))
                losses.append((-torch.log(pair_prob.clamp_min(1e-6)))[pair_mask].mean())

        if len(losses) == 0:
            return corr.new_zeros(())
        return torch.stack(losses).mean()


    def forward(self, image1, image2=None,  disp_init=None, sample=None):

        mode = 'stereo' if image2 is not None else 'mono'

        assert sample is not None

        with torch.no_grad():
            rel_depth_left = self.normalize(self.dfm(image1))
            rel_depth_right = self.normalize(self.dfm(image2)) if mode == 'stereo' else None

        grad_map = gen_x_grad_map(rel_depth_left.clone().requires_grad_(True))
        disp_modulation_factor = self.stereo_grad_modulator(grad_map.unsqueeze(1), rel_depth_left.unsqueeze(1))
        if mode == 'stereo' and rel_depth_right is not None:
            grad_map_right = gen_x_grad_map(rel_depth_right.clone().requires_grad_(True))
            disp_modulation_factor_right = self.stereo_grad_modulator(grad_map_right.unsqueeze(1), rel_depth_right.unsqueeze(1))
        else:
            grad_map_right = None
            disp_modulation_factor_right = None

        left_normal = sample['left_normal']  # Metric3D提供的法线图
        target_mask_left = sample['target_mask_left'] if 'target_mask_left' in sample else None
        target_mask_right = sample['target_mask_right'] if 'target_mask_right' in sample else None

        period = sample['period'] / (2 ** self.cfg['n_downsample']) if self.cfg['beat_repetitive_texture'] else None

        patch_h, patch_w = image1.shape[-2] // 14, image1.shape[-1] // 14

        # mono_features_left = self.encoder.get_intermediate_layers(image1, self.intermediate_layer_idx[self.encoder_name], return_class_token=True)
        # mono_features_right = self.encoder.get_intermediate_layers(image2, self.intermediate_layer_idx[self.encoder_name], return_class_token=True) \
            # if mode == 'stereo' else None
        
        # mono_features_left = self.readout_block(mono_features_left, patch_h, patch_w)
        # mono_features_right = self.readout_block(mono_features_right, patch_h, patch_w) \
        #     if mode == 'stereo' else None
        
        # mono_features_left = self.scratch_head(mono_features_left, patch_h, patch_w)
        # mono_features_right = self.scratch_head(mono_features_right, patch_h, patch_w) \
        #     if mode == 'stereo' else None
        
        # stereo_features_left = self.feat_transfer(mono_features_left)
        # stereo_features_right = self.feat_transfer(mono_features_right) \
        #     if mode == 'stereo' else None

        # 新特征读取
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            left_image_feats = sample.get('left_image_feats', None)
            right_image_feats = sample.get('right_image_feats', None)

            left_image_list = [np.array(_.cpu()) for _ in sample["left_image_raw"]]
            self.share_encoder.set_image_batch(left_image_list)
            left_mask_list = [np.array(_.cpu()) for _ in sample["target_mask_left"]]
            left_key_points_list = [get_kmeans_points(mask, 10) for mask in left_mask_list]
            left_key_points_list = [np.array(key_points) for key_points in left_key_points_list]
            left_image_feats = self.share_encoder.predict_batch(
                point_coords_batch=left_key_points_list,
                point_labels_batch=[np.ones(key_points.shape[0], dtype=int) for key_points in left_key_points_list],
                multimask_output=False            
            )
            if mode == 'stereo':
                right_image_list = [np.array(_.cpu()) for _ in sample["right_image_raw"]]
                self.share_encoder.set_image_batch(right_image_list)
                right_mask_list = [np.array(_.cpu()) for _ in sample["target_mask_right"]]
                right_key_points_list = [get_kmeans_points(mask, 10) for mask in right_mask_list]
                right_key_points_list = [np.array(key_points) for key_points in right_key_points_list]
                right_image_feats = self.share_encoder.predict_batch(
                    point_coords_batch=right_key_points_list,
                    point_labels_batch=[np.ones(key_points.shape[0], dtype=int) for key_points in right_key_points_list],
                    multimask_output=False            
                )
            else: right_image_feats = None

        sam_feat_sizes = get_sam_feat_sizes(image1.shape[-2], image1.shape[-1])

        if left_image_feats is not None:
            left_image_feats = [
                self.share_sam_feat_proj[i](
                    F.interpolate(
                        feat.squeeze(1).to(image1.device),
                        size=sam_feat_sizes[i],
                        mode='bilinear',
                        align_corners=False,
                    )
                )
                for i, feat in enumerate(left_image_feats)
            ]

        if mode == 'stereo' and right_image_feats is not None:
            right_image_feats = [
                self.share_sam_feat_proj[i](
                    F.interpolate(
                        feat.squeeze(1).to(image1.device),
                        size=sam_feat_sizes[i],
                        mode='bilinear',
                        align_corners=False,
                    )
                )
                for i, feat in enumerate(right_image_feats)
            ]
        
        mono_features_left = left_image_feats
        mono_features_right = right_image_feats if mode == 'stereo' else None


        """Task-aware feature transformation"""
        if mode == 'mono':
            mono_features_left = self.share_task_head(mono_features_left, encode_stereo=False)[0]
            mono_features_right, stereo_features_left, stereo_features_right = None, None, None
        else:
            mono_features_left, stereo_features_left = self.share_task_head(mono_features_left, encode_stereo=True)
            mono_features_right, stereo_features_right = self.share_task_head(mono_features_right, encode_stereo=True)


        match_left = F.normalize(self.stereo_desc(self.stereo_conv(stereo_features_left[0])), dim=1, eps=1e-6) if mode == 'stereo' else None
        match_right = F.normalize(self.stereo_desc(self.stereo_conv(stereo_features_right[0])), dim=1, eps=1e-6) if mode == 'stereo' else None
                
        if mode == 'stereo':
            context_features = self.stereo_context_ectractor(
                torch.cat((
                    image1, 
                    rel_depth_left.unsqueeze(1), 
                    left_normal,
                    grad_map.unsqueeze(1)
                    ), dim=1), 
                num_layers=self.cfg['n_gru_layers'])
            # context_features = self.context_ectractor(image1, num_layers=self.cfg['n_gru_layers'])

            net_list = [torch.tanh(x[0]) for x in context_features]  # Hidden state
            inp_list = [torch.relu(x[1]) for x in context_features]  # Conditioning input
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) 
                        for i,conv in zip(inp_list, self.stereo_context_zqr_convs)]
            
            corr_h, corr_w = net_list[0].shape[2], net_list[0].shape[3]
            match_left = F.interpolate(match_left, size=(corr_h, corr_w), mode='bilinear', align_corners=False)
            match_right = F.interpolate(match_right, size=(corr_h, corr_w), mode='bilinear', align_corners=False)

            # if self.cfg['use_roi_attn']:
            #     match_left = self.roi_cross_attn(
            #         match_left,
            #         target_mask=target_mask_left,
            #         target_bbox=None
            #     )
            #     match_right = self.roi_cross_attn(
            #         match_right,
            #         target_mask=target_mask_right,
            #         target_bbox=None
            #     )
                
            # 左右对称相关体与迭代：计算左视差和右视差序列
            corr_fn_left = corr_block(
                match_left,
                match_right,
                radius=self.cfg['corr_radius'],
                num_levels=self.cfg['corr_levels'],
                period_T=period
            )
            corr_fn_right = corr_block(
                match_right,
                match_left,
                radius=self.cfg['corr_radius'],
                num_levels=self.cfg['corr_levels'],
                period_T=period
            )

            # 左视图坐标初始化
            hor_coords0_L, hor_coords1_L = self.initialize_disp(net_list[0])

            # 右视图上下文与坐标初始化（对称）
            right_normal = sample.get('right_normal', None)
            context_features_r = self.stereo_context_ectractor(
                torch.cat((
                    image2,
                    rel_depth_right.unsqueeze(1),
                    right_normal,
                    grad_map_right.unsqueeze(1)
                ), dim=1),
                num_layers=self.cfg['n_gru_layers']
            ) if mode == 'stereo' else None

            if mode == 'stereo':
                net_list_r = [torch.tanh(x[0]) for x in context_features_r]
                inp_list_r = [torch.relu(x[1]) for x in context_features_r]
                inp_list_r = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i, conv in zip(inp_list_r, self.stereo_context_zqr_convs)]
                hor_coords0_R, hor_coords1_R = self.initialize_disp(net_list_r[0])
            else:
                net_list_r = inp_list_r = None
                hor_coords0_R = hor_coords1_R = None
            
            # 初始化扰动或给定初值
            if disp_init is not None:
                hor_coords1_L = hor_coords1_L + disp_init
                if mode == 'stereo':
                    # 右视图可以使用相反方向初值，或按需求调整
                    hor_coords1_R = hor_coords1_R + (-disp_init)
            else:
                disp_init_strategy = self.cfg.get('disp_init_strategy', 'random')
                if disp_init_strategy == 'relative_depth':
                    hor_coords1_L = hor_coords1_L + self.init_disp_from_rel_depth(rel_depth_left, hor_coords1_L)
                    if mode == 'stereo':
                        hor_coords1_R = hor_coords1_R + self.init_disp_from_rel_depth(rel_depth_right, hor_coords1_R)
                elif disp_init_strategy == 'random':
                    hor_coords1_L = hor_coords1_L + (torch.rand_like(hor_coords1_L)-0.5) * 2 * self.cfg['disp_init_var'] + self.cfg['relative_depth_disp_shift']
                    if mode == 'stereo':
                        hor_coords1_R = hor_coords1_R + (torch.rand_like(hor_coords1_R)-0.5) * 2 * self.cfg['disp_init_var'] + self.cfg['relative_depth_disp_shift']
                else:
                    raise ValueError(f"Unknown disp_init_strategy: {disp_init_strategy}")

            # 左向迭代
            disp_predictions_left = []
            cost_volume_ordinal_losses = []
            for itr in range(self.cfg['gru_iters']):
                hor_coords1_L = hor_coords1_L.detach()
                corr_L = corr_fn_left(hor_coords1_L)
                disp_L = hor_coords1_L - hor_coords0_L
                cost_volume_ordinal_losses.append(
                    self.cost_volume_ordinal_prior_loss(corr_L, disp_L, rel_depth_left)
                )
                with autocast(enabled=False):
                    if self.cfg['n_gru_layers'] == 3 and self.slow_fast_gru:
                        net_list = self.stereo_update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                    if self.cfg['n_gru_layers'] >= 2 and self.slow_fast_gru:
                        net_list = self.stereo_update_block(net_list, inp_list, iter32=self.cfg['n_gru_layers']==3, iter16=True, iter08=False, update=False)
                    net_list, up_mask_L, delta_disp_L = self.stereo_update_block(net_list, inp_list, corr_L, disp_L, iter32=self.cfg['n_gru_layers']==3, iter16=self.cfg['n_gru_layers']>=2)
                hor_coords1_L = hor_coords1_L + delta_disp_L
                disp_up_L = self.upsample_disp(hor_coords1_L - hor_coords0_L, up_mask_L)
                disp_up_L = F.interpolate(disp_up_L, size=image1.shape[2:], mode='bilinear', align_corners=True)
                if self.cfg['use_grad_modulator']:
                    disp_up_L = disp_up_L * (1 + disp_modulation_factor)
                disp_predictions_left.append(disp_up_L)

            # 右向迭代（对称计算）
            disp_predictions_right = []
            if mode == 'stereo':
                for itr in range(self.cfg['gru_iters']):
                    hor_coords1_R = hor_coords1_R.detach()
                    corr_R = corr_fn_right(hor_coords1_R)
                    disp_R = hor_coords1_R - hor_coords0_R
                    cost_volume_ordinal_losses.append(
                        self.cost_volume_ordinal_prior_loss(corr_R, disp_R, rel_depth_right)
                    )
                    with autocast(enabled=False):
                        if self.cfg['n_gru_layers'] == 3 and self.slow_fast_gru:
                            net_list_r = self.stereo_update_block(net_list_r, inp_list_r, iter32=True, iter16=False, iter08=False, update=False)
                        if self.cfg['n_gru_layers'] >= 2 and self.slow_fast_gru:
                            net_list_r = self.stereo_update_block(net_list_r, inp_list_r, iter32=self.cfg['n_gru_layers']==3, iter16=True, iter08=False, update=False)
                        net_list_r, up_mask_R, delta_disp_R = self.stereo_update_block(net_list_r, inp_list_r, corr_R, disp_R, iter32=self.cfg['n_gru_layers']==3, iter16=self.cfg['n_gru_layers']>=2)
                    hor_coords1_R = hor_coords1_R + delta_disp_R
                    disp_up_R = self.upsample_disp(hor_coords1_R - hor_coords0_R, up_mask_R)
                    disp_up_R = F.interpolate(disp_up_R, size=image2.shape[2:], mode='bilinear', align_corners=True)
                    if self.cfg['use_grad_modulator']:
                        disp_up_R = disp_up_R * (1 + disp_modulation_factor_right)
                    disp_predictions_right.append(disp_up_R)

            stereo_features = torch.cat([
                net_list[0],
                F.interpolate(net_list[1], size=(net_list[0].shape[2], net_list[0].shape[3]), mode='bilinear', align_corners=False)
            ], dim=1)

            # if random.random() < 0.5: stereo_features = None

        else:
            stereo_features = None
            cost_volume_ordinal_losses = []

        depth_prediction_L = self.mono_decoder(
            mono_features_left,
            patch_h, patch_w,
            condition_feat=stereo_features
        ) * self.max_depth

        depth_prediction_R = self.mono_decoder(
            mono_features_right,
            patch_h, patch_w,
            condition_feat= None  # 右视图默认无特征
        ) * self.max_depth if mode == 'stereo' else None

        if mode == 'stereo':
            scale_delt_L = image1.shape[-1] / disp_predictions_left[0].shape[-1]
            scale_delt_R = image2.shape[-1] / disp_predictions_right[0].shape[-1]
            disp_predictions_left = [scale_delt_L * d.squeeze(1) for d in disp_predictions_left]
            disp_predictions_right = [scale_delt_R * d.squeeze(1) for d in disp_predictions_right]
        else:
            disp_predictions_left = None
            disp_predictions_right = None

        return {
            'rel_depth_left': rel_depth_left,
            'rel_depth_right': rel_depth_right if mode == 'stereo' else None,
            'cost_volume_ordinal_loss': torch.stack(cost_volume_ordinal_losses).mean() if len(cost_volume_ordinal_losses) > 0 else image1.new_zeros(()),
            'depth_prediction_left': depth_prediction_L.squeeze(1),
            'depth_prediction_right': depth_prediction_R.squeeze(1) if mode == 'stereo' else None,
            'disp_predictions_left': disp_predictions_left,
            'disp_predictions_right': disp_predictions_right,
            'mono_features_left': mono_features_left,
            'mono_features_right': mono_features_right
        }
