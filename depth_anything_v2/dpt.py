import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .task_attn import TaskAttention


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)




class ReadoutBlock(nn.Module):  # 将其从DPTHead中分离出来
    def __init__(
        self, 
        in_channels, 
        out_channels=[48, 96, 192, 384], 
        use_clstoken=False
    ):
        super(ReadoutBlock, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)

        

        return out    



class DPTHead(nn.Module):
    def __init__(
        self, 
        features=256, 
        use_bn=False, 
        out_channels=[48, 96, 192, 384], 
    ):
        super(DPTHead, self).__init__()
        
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    
    def forward(self, out, patch_h, patch_w):
        
        layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = out
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out

class ScratchHead(nn.Module):
    def __init__(
        self,
        features=256,
        use_bn=False,
        out_channels=[48, 96, 192, 384],
    ): 
        super(ScratchHead, self).__init__()

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None

    def forward(self, out, patch_h, patch_w):
        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        return [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]



class ScratchHeadTaskAware(nn.Module):
    """
    替代 ScratchHead + Feat_transfer
    返回:
      mono_feats:   [4x, 8x, 16x, 32x] each C=features
      stereo_feats: [4x(48), 8x(64), 16x(192), 32x(160)] or None
    """
    def __init__(self, features=64, out_channels=[48, 96, 192, 384]):
        super().__init__()
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None

        self.task_attn = nn.ModuleList([TaskAttention(features) for _ in range(4)])

        # stereo pyramid head (替代原 feat_transfer)
        self.proj32 = nn.Sequential(
            nn.Conv2d(features, 160, 3, 1, 1),
            nn.InstanceNorm2d(160), nn.ReLU(True)
        )
        self.up32 = nn.ConvTranspose2d(160, 192, 3, 2, 1, output_padding=1, bias=False)

        self.fuse16 = nn.Sequential(
            nn.Conv2d(features + 192, 192, 5, 1, 2),
            nn.InstanceNorm2d(192), nn.ReLU(True)
        )
        self.res16 = nn.Conv2d(features, 192, 1)

        self.up16 = nn.ConvTranspose2d(192, 64, 3, 2, 1, output_padding=1, bias=False)
        self.fuse8 = nn.Sequential(
            nn.Conv2d(features + 64, 64, 5, 1, 2),
            nn.InstanceNorm2d(64), nn.ReLU(True)
        )
        self.res8 = nn.Conv2d(features, 64, 1)

        self.up8 = nn.ConvTranspose2d(64, 48, 3, 2, 1, output_padding=1, bias=False)
        self.fuse4 = nn.Sequential(
            nn.Conv2d(features + 48, 48, 5, 1, 2),
            nn.InstanceNorm2d(48), nn.ReLU(True)
        )
        self.res4 = nn.Conv2d(features, 48, 1)

    def forward(self, out, encode_stereo=True):
        l1, l2, l3, l4 = out
        l1 = self.scratch.layer1_rn(l1)
        l2 = self.scratch.layer2_rn(l2)
        l3 = self.scratch.layer3_rn(l3)
        l4 = self.scratch.layer4_rn(l4)

        # task_id=0: mono depth
        d1 = self.task_attn[0](l1, 0)
        d2 = self.task_attn[1](l2, 0)
        d3 = self.task_attn[2](l3, 0)
        d4 = self.task_attn[3](l4, 0)
        mono_feats = [d1, d2, d3, d4]

        if not encode_stereo:
            return mono_feats, None

        # task_id=1: stereo match
        s1 = self.task_attn[0](l1, 1)
        s2 = self.task_attn[1](l2, 1)
        s3 = self.task_attn[2](l3, 1)
        s4 = self.task_attn[3](l4, 1)

        f32 = self.proj32(s4)
        f32_up = self.up32(f32)
        if f32_up.shape[-2:] != s3.shape[-2:]:
            f32_up = F.interpolate(f32_up, size=s3.shape[-2:], mode="bilinear", align_corners=False)

        f16 = self.fuse16(torch.cat([s3, f32_up], dim=1)) + self.res16(s3)
        f16_up = self.up16(f16)
        if f16_up.shape[-2:] != s2.shape[-2:]:
            f16_up = F.interpolate(f16_up, size=s2.shape[-2:], mode="bilinear", align_corners=False)

        f8 = self.fuse8(torch.cat([s2, f16_up], dim=1)) + self.res8(s2)
        f8_up = self.up8(f8)
        if f8_up.shape[-2:] != s1.shape[-2:]:
            f8_up = F.interpolate(f8_up, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        f4 = self.fuse4(torch.cat([s1, f8_up], dim=1)) + self.res4(s1)

        stereo_feats = [f4, f8, f16, f32]
        return mono_feats, stereo_feats
   


class DPTHead_FiLM(nn.Module):
    def __init__(
        self,
        features=256,
        use_bn=False,
        out_channels=[48, 96, 192, 384],
        film_channels=None,  # 新增：FiLM 条件特征通道数（默认与 features 相同）
    ):
        super(DPTHead_FiLM, self).__init__()

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # FiLM 调制器：将条件特征映射为每通道的 gamma / beta
        self.film_channels = film_channels or features
        self.film_modulator = nn.Sequential(
            nn.Conv2d(self.film_channels, features, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(features, features * 2, kernel_size=1),
        )

    def forward(self, out, patch_h, patch_w, condition_feat=None):
        layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = out

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # FiLM 融合（可选）
        if condition_feat is not None:
            if condition_feat.dim() == 2:
                film_feat_map = condition_feat[:, :, None, None]
            elif condition_feat.dim() == 3:
                film_feat_map = condition_feat.unsqueeze(-1)
            else:
                film_feat_map = condition_feat

            if film_feat_map.dim() == 4 and film_feat_map.shape[2:] != path_1.shape[2:]:
                film_feat_map = F.interpolate(
                    film_feat_map,
                    size=path_1.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            modulation = self.film_modulator(film_feat_map)
            gamma, beta = modulation.chunk(2, dim=1)
            path_1 = path_1 * (1 + gamma) + beta  # FiLM: y = (1+gamma) * x + beta

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out