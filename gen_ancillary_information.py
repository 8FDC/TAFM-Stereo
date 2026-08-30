import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
from glob import glob
import os
from tqdm import tqdm

from yolo_sam.yolo11.ultralytics import YOLO
from yolo_sam.sam2.build_sam import build_sam2
from yolo_sam.crab_segment import crab_segmentor
from Metric3D.hubconf import metric3d_vit_small


def gen_ancillary_information(image_path, models, detect_target=False, predict_normal=False):
    """
    sam + yolo -> 目标分割掩码
    metric3d -> 法向量
    """
    
    yolo = models['yolo']
    sam = models['sam']
    metric3d = models['metric3d']

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    if detect_target:
        target_bbox, target_mask = crab_segmentor(image, yolo, sam)
        target_mask = target_mask.cpu().numpy()
        target_bbox = target_bbox.cpu().numpy()
        cv2.imwrite(
            image_path.replace('.png', '_target_mask.png'), 
            target_mask.astype(np.uint8)*255
            )
        cv2.imwrite(
            image_path.replace('.png', '_target_bbox.png'),
            target_bbox.astype(np.uint8)*255 
            )
    if predict_normal:
        image_ = torch.from_numpy(image/255.0).permute(2, 0, 1).unsqueeze(0).float().cuda()
        _, _, output_dict = metric3d.inference({"input": image_})
        normal = output_dict['prediction_normal'].squeeze()[:3, :, :].permute(1, 2, 0).cpu().numpy()

    if not os.path.exists(os.path.dirname(image_path.replace('image', 'normal'))):
        os.makedirs(os.path.dirname(image_path.replace('image', 'normal')))

    cv2.imwrite(
        image_path.replace('image', 'normal'), 
        ((normal+1)/2 * 255).astype(np.uint8)
        )
    
    return None


if __name__ == '__main__':
    
    device = torch.device('cuda:0')

    metric3d = metric3d_vit_small(pretrain=False)
    metric3d.load_state_dict(
        torch.load("Metric3D/checkpoints/metric_depth_vit_small_800k.pth")['model_state_dict'],
        strict=False
    )

    models = {
        'yolo': YOLO('yolo_sam/yolo11/checkpoints/best.pt').to(device).eval(),
        
        'sam': build_sam2(
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "yolo_sam/checkpoints/sam2.1_hiera_base_plus.pt",
            device=device
        ).to(device).eval(),
        
        'metric3d': metric3d.to(device).eval(),
    }

    # 
    image_paths = glob(rf'F:\dataset\stereo_matching_dataset\*\*\color_right_*.jpg')
    image_paths.sort()


    for image_path in tqdm(image_paths):
        gen_ancillary_information(image_path, models, predict_normal=True)
