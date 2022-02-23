import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb


# This is a PyTorch data augmentation library, that takes PyTorch Tensor as input
# Functions can be applied in the __getitem__ function to do augmentation on the fly during training.
# These functions can be easily parallelized by setting 'num_workers' in pytorch dataloader.

# tensor_img: 1, C, (D), H, W

def gaussian_noise(tensor_img, std, mean=0):
    
    return tensor_img + torch.randn(tensor_img.shape).to(tensor_img.device) * std + mean

def brightness_additive(tensor_img, std, mean=0, per_channel=False):
    
    if per_channel:
        C = tensor_img.shape[1]
    else:
        C = 1

    if len(tensor_img.shape) == 5:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1, 1)).to(tensor_img.device)
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1)).to(tensor_img.device)
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img + rand_brightness

def gamma(tensor_img, gamma_range=(0.5, 2), per_channel=False, retain_stats=False):
    
    if len(tensor_img.shape) == 5:
        dim = '3d'
        _, C, D, H, W = tensor_img.shape
    elif len(tensor_img.shape) == 4:
        dim = '2d'
        _, C, H, W = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')
    
    tmp_C = C if per_channel else 1
    
    tensor_img = tensor_img.view(tmp_C, -1)
    minm, _ = tensor_img.min(dim=1)
    maxm, _ = tensor_img.max(dim=1)
    minm, maxm = minm.unsqueeze(1), maxm.unsqueeze(1) # unsqueeze for broadcast machanism

    rng = maxm - minm

    mean = tensor_img.mean(dim=1).unsqueeze(1)
    std = tensor_img.std(dim=1).unsqueeze(1)
    gamma = torch.rand(C, 1) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]

    tensor_img = torch.pow((tensor_img - minm) / rng, gamma) * rng + minm

    if retain_stats:
        tensor_img -= tensor_img.mean(dim=1).unsqueeze(1)
        tensor_img = tensor_img / tensor_img.std(dim=1).unsqueeze(1) * std + mean

    if dim == '3d':
        return tensor_img.view(1, C, D, H, W)
    else:
        return tensor_img.view(1, C, H, W)
        
def random_scale_rotate_translate_2d(tensor_img, tensor_lab, scale, rotate, translate):

    # implemented with affine transformation

    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 2
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 2
    

    scale_x = 1 - scale[0] + np.random.random() * 2*scale[0]
    scale_y = 1 - scale[1] + np.random.random() * 2*scale[1]
    shear_x = np.random.random() * 2*scale[0] - scale[0] 
    shear_y = np.random.random() * 2*scale[1] - scale[1]
    translate_x = np.random.random() * 2*translate[0] - translate[0]
    translate_y = np.random.random() * 2*translate[1] - translate[1]

    theta_scale = torch.tensor([[scale_x, shear_x, translate_x], 
                                [shear_y, scale_y, translate_y],
                                [0, 0, 1]]).float()
    angle = (float(np.random.randint(-rotate, max(rotate, 1))) / 180.) * math.pi

    theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle), math.cos(angle), 0],
                                [0, 0, 1]]).float()
    
    theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
    grid = F.affine_grid(theta.unsqueeze(0), tensor_img.size(), align_corners=True)

    tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()

    return tensor_img, tensor_lab


def random_scale_rotate_translate_3d(tensor_img, tensor_lab, scale, rotate, translate, noshear=True):
    
    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 3
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 3
    if isinstance(rotate, float) or isinstance(rotate, int):
        rotate = [rotate] * 3

    scale_z = 1 - scale[0] + np.random.random() * 2*scale[0]
    scale_x = 1 - scale[1] + np.random.random() * 2*scale[1]
    scale_y = 1 - scale[2] + np.random.random() * 2*scale[2]
    shear_zx = 0 if noshear else np.random.random() * 2*scale[0] - scale[0]
    shear_zy = 0 if noshear else np.random.random() * 2*scale[0] - scale[0]
    shear_xz = 0 if noshear else np.random.random() * 2*scale[1] - scale[1]
    shear_xy = 0 if noshear else np.random.random() * 2*scale[1] - scale[1]
    shear_yz = 0 if noshear else np.random.random() * 2*scale[2] - scale[2]
    shear_yx = 0 if noshear else np.random.random() * 2*scale[2] - scale[2]
    translate_z = np.random.random() * 2*translate[0] - translate[0]
    translate_x = np.random.random() * 2*translate[1] - translate[1]
    translate_y = np.random.random() * 2*translate[2] - translate[2]

    theta_scale = torch.tensor([[scale_z, shear_zx, shear_zy, translate_z],
                                [shear_xz, scale_x, shear_xy, translate_x],
                                [shear_yz, shear_yx, scale_y, translate_y], 
                                [0, 0, 0, 1]]).float()

    angle_xy = (float(np.random.randint(-rotate[0], max(rotate[0], 1))) / 180.) * math.pi
    angle_xz = (float(np.random.randint(-rotate[1], max(rotate[1], 1))) / 180.) * math.pi
    angle_yz = (float(np.random.randint(-rotate[2], max(rotate[2], 1))) / 180.) * math.pi
    
    theta_rotate_xy = torch.tensor([[1, 0, 0, 0],
                                    [0, math.cos(angle_xy), -math.sin(angle_xy), 0],
                                    [0, math.sin(angle_xy), math.cos(angle_xy), 0],
                                    [0, 0, 0, 1]]).float()
    theta_rotate_xz = torch.tensor([[math.cos(angle_xz), -math.sin(angle_xz), 0, 0],
                                    [math.sin(angle_xz), math.cos(angle_xz), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]).float()
    theta_rotate_yz = torch.tensor([[math.cos(angle_yz), 0, -math.sin(angle_yz), 0],
                                    [0, 1, 0, 0],
                                    [math.sin(angle_yz), 0, math.cos(angle_yz), 0],
                                    [0, 0, 0, 1]]).float()

    theta = torch.mm(theta_rotate_xy, theta_rotate_xz)
    theta = torch.mm(theta, theta_rotate_yz)
    theta = torch.mm(theta, theta_scale)[0:3, :].unsqueeze(0)
    

    grid = F.affine_grid(theta, tensor_img.size(), align_corners=True)
    tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()

    return tensor_img, tensor_lab
    

def crop_2d(tensor_img, tensor_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 2

    _, _, H, W = tensor_img.shape

    diff_H = H - crop_size[0]
    diff_W = W - crop_size[1]
    
    if mode == 'random':
        rand_x = np.random.randint(0, max(diff_H, 1))
        rand_y = np.random.randint(0, max(diff_W, 1))
    else:
        rand_x = diff_H // 2
        rand_y = diff_W // 2

    cropped_img = tensor_img[:, :, rand_x:rand_x+crop_size[0], rand_y:rand_y+crop_size[1]]
    cropped_lab = tensor_lab[:, :, rand_x:rand_x+crop_size[0], rand_y:rand_y+crop_size[1]]

    return cropped_img, cropped_lab


def crop_3d(tensor_img, tensor_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    _, _, D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    if mode == 'random':
        rand_z = np.random.randint(0, max(diff_D, 1))
        rand_x = np.random.randint(0, max(diff_H, 1))
        rand_y = np.random.randint(0, max(diff_W, 1))
    else:
        rand_z = diff_D // 2
        rand_x = diff_H // 2
        rand_y = diff_W // 2

    cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_x:rand_x+crop_size[1], rand_y:rand_y+crop_size[2]]
    cropped_lab = tensor_lab[:, :, rand_z:rand_z+crop_size[0], rand_x:rand_x+crop_size[1], rand_y:rand_y+crop_size[2]]

    return cropped_img, cropped_lab

