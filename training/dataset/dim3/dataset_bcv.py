import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation

class BCVDataset(Dataset):
    def __init__(self, args, mode='train', k_fold=5, k=0, seed=0):
        
        self.mode = mode
        self.args = args

        assert mode in ['train', 'test']

        with open(os.path.join(args.data_root, 'list', 'dataset.yaml'), 'r') as f:
            img_name_list = yaml.load(f, Loader=yaml.SafeLoader)


        random.Random(seed).shuffle(img_name_list)

        length = len(img_name_list)
        test_name_list = img_name_list[k*(length//k_fold) : (k+1)*(length//k_fold)]
        train_name_list = list(set(img_name_list) - set(test_name_list))
        
        if mode == 'train':
            img_name_list = train_name_list
        else:
            img_name_list = test_name_list

        print(img_name_list)
        print('Start loading %s data'%self.mode)

        path = args.data_root

        self.img_list = []
        self.lab_list = []
        self.spacing_list = []

        for name in img_name_list:
                
            img_name = name + '.nii.gz'
            lab_name = name + '_gt.nii.gz'

            itk_img = sitk.ReadImage(os.path.join(path, img_name))
            itk_lab = sitk.ReadImage(os.path.join(path, lab_name))

            spacing = np.array(itk_lab.GetSpacing()).tolist()
            self.spacing_list.append(spacing[::-1])  # itk axis order is inverse of numpy axis order

            assert itk_img.GetSize() == itk_lab.GetSize()

            img, lab = self.preprocess(itk_img, itk_lab)

            self.img_list.append(img)
            self.lab_list.append(lab)

        
        print('Load done, length of dataset:', len(self.img_list))

    def __len__(self):
        if self.mode == 'train':
            return len(self.img_list) * 100000
        else:
            return len(self.img_list)

    def preprocess(self, itk_img, itk_lab):
        
        img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        lab = sitk.GetArrayFromImage(itk_lab).astype(np.uint8)

        img = np.clip(img, -958, 327)
        img -= 82.92
        img /= 136.97

        z, y, x = img.shape
        
        # pad if the image size is smaller than trainig size
        if z < self.args.training_size[0]:
            diff = int(math.ceil((self.args.training_size[0] - z) / 2))
            img = np.pad(img, ((diff, diff), (0,0), (0,0)))
            lab = np.pad(lab, ((diff, diff), (0,0), (0,0)))
        if y < self.args.training_size[1]:
            diff = int(math.ceil((self.args.training_size[1]+2 - y) / 2))
            img = np.pad(img, ((0,0), (diff,diff), (0,0)))
            lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))
        if x < self.args.training_size[2]:
            diff = int(math.ceil((self.args.training_size[2]+2 - x) / 2))
            img = np.pad(img, ((0,0), (0,0), (diff, diff)))
            lab = np.pad(lab, ((0,0), (0,0), (diff, diff)))

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        assert tensor_img.shape == tensor_lab.shape
        
        return tensor_img, tensor_lab

    def __getitem__(self, idx):
        
        idx = idx % len(self.img_list)
        
        tensor_img = self.img_list[idx]
        tensor_lab = self.lab_list[idx]

        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
        tensor_lab = tensor_lab.unsqueeze(0).unsqueeze(0)
        # 1, C, D, H, W


        if self.mode == 'train':
            d, h, w = self.args.training_size
            
            # Gaussian Noise
            tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)

            
            if np.random.random() < 0.5:

                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d+15, h+65, w+65], mode='random')
                tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
            
            else:
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')


        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape == tensor_lab.shape

        if self.mode == 'train':
            return tensor_img, tensor_lab
        else:
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])

            
