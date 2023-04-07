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
import os

class AMOSDataset(Dataset):
    def __init__(self, args, mode='train', k_fold=5, k=0, seed=0):
        
        self.mode = mode
        self.args = args

        assert mode in ['train', 'test']

        with open(os.path.join(args.data_root, 'list', 'dataset.yaml'), 'r') as f:
            img_name_list = yaml.load(f, Loader=yaml.SafeLoader)


        random.Random(seed).shuffle(img_name_list)

        #length = len(img_name_list)
        #test_name_list = img_name_list[k*(length//k_fold) : (k+1)*(length//k_fold)]
        #train_name_list = list(set(img_name_list) - set(test_name_list))
        
        if mode == 'train':
            pass
        else:
            #img_name_list = test_name_list
            img_name_list = [13, 70, 292, 280, 29, 334, 257, 357, 326, 191, 238, 310, 373, 202, 247, 255, 228, 328, 363, 200, 56, 144, 290, 308, 208, 316, 216, 204, 304, 85, 189, 140, 40, 123, 286, 176, 284, 150, 117, 174, 206, 218, 318, 365, 377, 87, 372, 311, 203, 356, 339, 244, 344, 90, 293, 128, 155, 136, 63, 112, 34, 283, 157, 73, 61, 313, 325, 258, 409, 346, 106, 18, 22, 41, 287, 399, 333, 233, 250, 342, 309, 278, 223, 323, 194, 352, 364, 219, 207, 368, 8, 108, 167, 51, 132, 385, 32, 289, 397, 120]

        print('Start loading %s data'%self.mode)
        print(img_name_list)
        path = args.data_root

        self.img_list = []
        self.lab_list = []
        self.spacing_list = []

        for name in img_name_list:
            img_name = '%d.nii.gz'%name
            lab_name = '%d_gt.nii.gz'%name

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

        img = np.clip(img, -991, 362)
        img -= 50.0
        img /= 141.0

        z, y, x = img.shape
        
        # pad if the image size is smaller than trainig size
        if self.args.dataloader == 'dali':
            pad_size = [i+j for i,j in zip(self.args.training_size, self.args.affine_pad_size)]
        else:
            pad_size = self.args.training_size

        if z < pad_size[0]:
            diff = int(math.ceil((pad_size[0] - z) / 2))
            img = np.pad(img, ((diff, diff), (0,0), (0,0)))
            lab = np.pad(lab, ((diff, diff), (0,0), (0,0)))
        if y < pad_size[1]:
            diff = int(math.ceil((pad_size[1]+2 - y) / 2))
            img = np.pad(img, ((0,0), (diff,diff), (0,0)))
            lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))
        if x < pad_size[2]:
            diff = int(math.ceil((pad_size[2]+2 - x) / 2))
            img = np.pad(img, ((0,0), (0,0), (diff, diff)))
            lab = np.pad(lab, ((0,0), (0,0), (diff, diff)))

        tensor_img = torch.from_numpy(img).to(torch.float)
        tensor_lab = torch.from_numpy(lab).to(torch.int8)

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
            
            _, _, d, h, w = tensor_img.shape
            
            if np.random.random() < 0.5:
                # crop trick for faster augmentation
                # crop a sub volume for scaling and rotation
                # instead of scaling and rotating the whole image
                crop_size = [min(i+j, k) for i,j,k in zip(self.args.training_size, self.args.affine_pad_size, [d, h, w])]
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, crop_size, mode='random')
                if self.args.aug_device == 'gpu':
                    tensor_img = tensor_img.cuda(self.args.proc_idx)
                    tensor_lab = tensor_lab.cuda(self.args.proc_idx)

                tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
            else:
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')
                if self.args.aug_device == 'gpu':
                    tensor_img = tensor_img.cuda(self.args.proc_idx)
                    tensor_lab = tensor_lab.cuda(self.args.proc_idx)

            tensor_img, tensor_lab = tensor_img.contiguous(), tensor_lab.contiguous()

            if np.random.random() < 0.2:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
            if np.random.random() < 0.2:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            if np.random.random() < 0.2:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
            if np.random.random() < 0.2:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
            if np.random.random() < 0.2:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
            if np.random.random() < 0.2:
                std = np.random.random() * 0.1
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)

             

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape == tensor_lab.shape

        if self.mode == 'train':
            return tensor_img, tensor_lab
        else:
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])

         
