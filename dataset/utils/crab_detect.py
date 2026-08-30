import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def gen_crab_mask(image, model):
    
    boxes = model(image, verbose=False)[0].boxes.xyxy

    crab_mask = torch.zeros(image.shape[:2]).numpy().astype(np.uint8)

    boxes = [tuple(map(int, line)) for line in boxes]

    for box in boxes:
        x1, y1, x2, y2 = box
        crab_mask[y1:y2, x1:x2] = 1
    
    return crab_mask, boxes



def main():
    
    image = cv2.cvtColor(cv2.imread("assets/0002.jpg"), cv2.COLOR_RGB2BGR)
    crab_mask = gen_crab_mask(image)

    pass