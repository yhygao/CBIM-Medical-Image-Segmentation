import builtins
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from inference.utils import get_inference
from dataset_conversion.utils import ResampleXYZAxis, ResampleLabelToRef
from torch.utils import data

import SimpleITK as sitk
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings

import matplotlib.pyplot as plt

from utils import (
    configure_logger,
    save_configure,
)
warnings.filterwarnings("ignore", category=UserWarning)



def prediction(model_list, tensor_img, args):
    
    inference = get_inference(args)

    with torch.no_grad():
        tensor_img = tensor_img.cuda().float()
        D, H, W = tensor_img.shape
        tensor_pred = torch.zeros([args.classes, D, H, W]).to(tensor_img.device)
        
        if args.dimension == '2d':
            tensor_img = tensor_img.unsqueeze(0).permute(1, 0, 2, 3)
        else:
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
        
        for model in model_list:
            pred = inference(model, tensor_img, args)
            
            if args.dimension == '2d':
                pred = pred.permute(1, 0, 2, 3)
            else:
                pred = pred.squeeze(0)
            
            tensor_pred += pred       
       
        _, label_pred = torch.max(tensor_pred, dim=0)


    return label_pred


def pad_to_training_size(np_img, args):

    z, y, x = np_img.shape
   
    if args.dimension == '3d':
        if z < args.training_size[0]:
            diff = (args.training_size[0]+2 - z) // 2
            np_img = np.pad(np_img, ((diff, diff), (0,0), (0,0)))
            z_start = diff
            z_end = diff + z
        else:
            z_start = 0
            z_end = z

        if y < args.training_size[1]:
            diff = (args.training_size[1]+2 - y) // 2
            np_img = np.pad(np_img, ((0,0), (diff, diff), (0,0)))
            y_start = diff
            y_end = diff + y
        else:
            y_start = 0
            y_end = y

        if x < args.training_size[2]:
            diff = (args.training_size[2]+2 -x) // 2
            np_img = np.pad(np_img, ((0,0), (0,0), (diff, diff)))
            x_start = diff
            x_end = diff + x
        else:
            x_start = 0
            x_end = x

        return np_img, [z_start, z_end, y_start, y_end, x_start, x_end]

    elif args.dimension == '2d':
        
        if y < args.training_size[0]:
            diff = (args.training_size[0]+2 - y) // 2
            np_img = np.pad(np_img, ((0,0), (diff, diff), (0,0)))
            y_start = diff
            y_end = diff + y
        else:
            y_start = 0
            y_end = y

        if x < args.training_size[1]:
            diff = (args.training_size[1]+2 -x) // 2
            np_img = np.pad(np_img, ((0,0), (0,0), (diff, diff)))
            x_start = diff
            x_end = diff + x
        else:
            x_start = 0
            x_end = x

        return np_img, [y_start, y_end, x_start, x_end]

    else:
        raise ValueError




def unpad_img(np_pred, original_idx, args):
    if args.dimension == '3d':
        z_start, z_end, y_start, y_end, x_start, x_end = original_idx
    
        return np_pred[z_start:z_end, y_start:y_end, x_start:x_end]
    elif args.dimension == '2d':
        y_start, y_end, x_start, x_end = original_idx

        return np_pred[:, y_start:y_end, x_start:x_end]
        
    else:
        raise ValueError


def preprocess(itk_img, target_spacing, args):
    '''
    This function performs preprocessing to make images to be consistent with training, e.g. spacing resample, redirection and etc.
    Args:
        itk_img: the simpleITK image to be predicted
    Return: the preprocessed image tensor
    '''
   
    if args.dimension == '3d':
        # target spacing: (x, y, z)
        if itk_img.GetSpacing() != target_spacing:
            itk_img = ResampleXYZAxis(itk_img, target_spacing, interp=sitk.sitkBSpline)
    elif args.dimension == '2d':
        # current only support resize x,y, and keep z unchanged
        x_spacing, y_spacing, z_spacing = itk_img.GetSpacing()
        if (x_spacing, y_spacing) != target_spacing:
            itk_img = ResampleXYZAxis(itk_img, (target_spacing[0], target_spacing[1], z_spacing), interp=sitk.sitkBSpline)
    else:
        raise ValueError
        
    np_img = sitk.GetArrayFromImage(itk_img)
    
    '''
    Need to modify the following preprocessing steps to be consistent with training. Copy from the dataset_xxx.py
    '''
    #np_img = np.clip(np_img, -79, 304)
    #np_img -= 100.93
    #np_img /= 76.90
    max98 = np.percentile(np_img, 98)
    np_img = np.clip(np_img, 0, 98)
    np_img = np_img / max98

    np_img, original_idx = pad_to_training_size(np_img, args)

    tensor_img = torch.from_numpy(np_img)

    return tensor_img, original_idx


def postprocess(tensor_pred, itk_img, original_idx, args):
    np_pred = tensor_pred.cpu().numpy().astype(np.uint8)

    np_pred = unpad_img(np_pred, original_idx, args)

    itk_pred = sitk.GetImageFromArray(np_pred)
    if args.dimension == '2d':
        target_spacing = (args.target_spacing[0], args.target_spacing[1], itk_img.GetSpacing()[2])
    else:
        target_spacing = args.target_spacing

    if target_spacing != itk_img.GetSpacing():
        itk_pred.SetSpacing(target_spacing)
        itk_pred.SetOrigin(itk_img.GetOrigin())
        itk_pred.SetDirection(itk_img.GetDirection())
        itk_pred = ResampleLabelToRef(itk_pred, itk_img, interp=sitk.sitkNearestNeighbor)

    itk_pred.CopyInformation(itk_img)

    return itk_pred




def init_model(args):

    model_list = []
    for ckp_path in args.load:
        model = get_model(args)
        pth = torch.load(ckp_path, map_location=torch.device('cpu'))
        
        if args.ema:
            model.load_state_dict(pth['ema_model_state_dict'])
        else:
            model.load_state_dict(pth['model_state_dict'])
        
        # If you want to load checkpoint trained with previous version code, use the following instead:
        #model.load_state_dict(pth)


        model.cuda()
        model_list.append(model)
        print(f"Model loaded from {ckp_path}")

    return model_list


def get_parser():

    def parse_spacing_list(string):
        return tuple([float(spacing) for spacing in string.split(',')])
    def parse_model_list(string):
        return string.split(',')
    parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
    parser.add_argument('--dataset', type=str, default='kits', help='dataset name')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')

    parser.add_argument('--load', type=parse_model_list, default=False, help='the path of trained model checkpoint. Use \',\' as the separator if load multiple checkpoints for ensemble')
    parser.add_argument('--img_path', type=str, default=False, help='the path of the directory of images to be predicted')
    parser.add_argument('--save_path', type=str, default='./result/', help='the path to save predicted label')
    parser.add_argument('--target_spacing', type=parse_spacing_list, default='1.0,1.0,1.0', help='the spacing that used for training, in x,y,z order for 3d, and x,y order for 2d')
    
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args
    



if __name__ == '__main__':
    
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.sliding_window = True
    args.window_size = args.training_size


    model_list = init_model(args)

    for img_name in os.listdir(args.img_path):
        
        itk_img = sitk.ReadImage(os.path.join(args.img_path, img_name))
        
        tmp_itk_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(itk_img))
        tmp_itk_img.CopyInformation(itk_img)
        
        tensor_img, original_idx = preprocess(tmp_itk_img, args.target_spacing, args)
        pred_label = prediction(model_list, tensor_img, args)
        itk_pred = postprocess(pred_label, itk_img, original_idx, args)

        sitk.WriteImage(itk_pred, os.path.join(args.save_path, img_name))

        print(img_name, 'done')


    




