# load pictures and save using 2 functions to compare the difference
# between the two functions

import matplotlib.pyplot as plt

import torchvision.utils as vutils
from PIL import Image
import numpy as np
import os
import torch
import timm

# load pictures
img = Image.open('/opt/zhanghx/divit/generations/vit_inversion/best_images/output_00017_gpu_0.png')
img = np.array(img)
img = torch.from_numpy(img)

def get_mean_std(model_name, use_fp16=False):
    if 'vit' in model_name:
        model = timm.create_model(model_name, pretrained=True)
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    if use_fp16:
        mean = np.array(mean, dtype=np.float16)
        std = np.array(std, dtype=np.float16)
    return mean, std

def clip(image_tensor, model_name, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    mean, std = get_mean_std(model_name, use_fp16)

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def denormalize(image_tensor, model_name, use_fp16=False):
    '''
    convert floats back to input
    '''
    mean, std = get_mean_std(model_name, use_fp16)

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

# save pictures using torchvision.utils
img = clip(img, 'vit_base_patch16_224')

vutils.save_image(img, '/opt/zhanghx/divit/generations/vit_inversion/best_images/output_00017_gpu_0_test1.jpg', normalize=True, scale_each=True, nrow=int(1))
best_inputs = denormalize(img, 'vit_base_patch16_224', use_fp16=False)
image_np = img.data.cpu().numpy().transpose((1, 2, 0))
pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
pil_image.save('/opt/zhanghx/divit/generations/vit_inversion/best_images/output_00017_gpu_0_test2.png')
