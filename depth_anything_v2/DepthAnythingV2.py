import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose


from .dinov2 import DINOv2
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .dpt import DPTHead, ReadoutBlock, ScratchHead


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder_name='vits', 
        features=256, 
        out_channels=[48, 96, 192, 384], 
        use_bn=False, 
        use_clstoken=False,
        max_depth=550.0
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.max_depth = max_depth
        
        self.encoder_name = encoder_name
        self.encoder = DINOv2(model_name=encoder_name)
        
        self.readout_block = ReadoutBlock(self.encoder.embed_dim, out_channels=out_channels, use_clstoken=use_clstoken)
        self.scratch_head = ScratchHead(features, use_bn, out_channels=out_channels)
        self.decoder = DPTHead(features, use_bn, out_channels=out_channels)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.encoder.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder_name], return_class_token=True)
        features = self.readout_block(features, patch_h, patch_w)  # 将patch特征转换为适合深度头输入的特征
        features = self.scratch_head(features, patch_h, patch_w)  # 进一步处理特征以适应深度头
        depth = self.decoder(features, patch_h, patch_w) * self.max_depth
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)


