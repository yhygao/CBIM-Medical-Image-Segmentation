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
from training import augmentation, augmentation_dali
from training.dataset.utils import DALIInputCallable
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
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
            #img_name_list = train_name_list[:4]
            pass
        else:
            #img_name_list = test_name_list[:4]
            img_name_list = [553, 575, 598, 559, 547, 563, 549, 545, 573, 561, 552, 568, 576, 550, 562, 546, 572, 556, 544, 581]

        
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
        
        percentile_2 = np.percentile(img, 2, axis=None)
        percentile_98 = np.percentile(img, 98, axis=None)

        img = np.clip(img, percentile_2, percentile_98)
        mean = np.mean(img)
        std = np.std(img)
        img -= mean
        img /= std

        z, y, x = img.shape
        
        # pad if the image size is smaller than trainig+pad size
        pad_size = [i+j for i,j in zip(self.args.training_size, self.args.affine_pad_size)]

        if z < pad_size[0]:
            diff = int(math.ceil((pad_size[0] - z) / 2))
            img = np.pad(img, ((diff, diff), (0,0), (0,0)))
            lab = np.pad(lab, ((diff, diff), (0,0), (0,0)))
        if y < pad_size[1]:
            diff = int(math.ceil((pad_size[1] - y) / 2))
            img = np.pad(img, ((0,0), (diff,diff), (0,0)))
            lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))
        if x < pad_size[2]:
            diff = int(math.ceil((pad_size[2] - x) / 2))
            img = np.pad(img, ((0,0), (0,0), (diff, diff)))
            lab = np.pad(lab, ((0,0), (0,0), (diff, diff)))

        tensor_img = torch.from_numpy(img).float()
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
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx)
                tensor_lab = tensor_lab.cuda(self.args.proc_idx)
            
            d, h, w = self.args.training_size


            if np.random.random() < 0.2:
                # crop trick for faster augmentation
                # crop a sub volume for scaling and rotation
                # instead of scaling and rotating the whole image
                pad_size = [i+j for i,j in zip(self.args.training_size, self.args.affine_pad_size)]
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, pad_size, mode='random')
                tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
            else:
                 tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')
            tensor_img, tensor_lab = tensor_img.contiguous(), tensor_lab.contiguous()

            # Gaussian Noise
            if np.random.random() < 0.2: 
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
            if np.random.random() < 0.2:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.2)
            if np.random.random() < 0.2:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
            if np.random.random() < 0.2:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
            if np.random.random() < 0.2:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.0])
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


    def getitem_dali(self, idx):
        
        #print(self.args.proc_idx, os.getpid(), idx)
        idx = idx % len(self.img_list)
        
        tensor_img = self.img_list[idx]
        tensor_lab = self.lab_list[idx]


        tensor_img = tensor_img.unsqueeze(3).float() # DHWC
        tensor_lab = tensor_lab.unsqueeze(3).to(torch.int32) # DHWC
        # cuda or not depends on the device, if gpu, and parallel is true, then no_copy is true, need to be cuda
        # else is fine on cpu
        if self.args.aug_device == 'cpu':
            return tensor_img, tensor_lab
        elif self.args.aug_device == 'gpu':
            return tensor_img.cuda(self.args.proc_idx), tensor_lab.cuda(self.args.proc_idx)
    
    #@staticmethod
    @pipeline_def
    def dali_pipeline(self, dataset, bs, device='cpu', shard_id=0, num_shards=1):
        # need to add gamma, and pass arguments into affine
        img, lab = fn.external_source(source=DALIInputCallable(dataset, bs, shard_id, num_shards), num_outputs=2, batch=False,
                layout=['DHWC', 'DHWC'], dtype=[types.FLOAT, types.INT32], parallel=True, device=device)
        img = augmentation_dali.brightness(img, additive_range=(-0.1, 0.1), multiply_range=(0.7, 1.3), p=0.2)
        img = augmentation_dali.contrast(img, contrast_range=(0.65, 1.5), p=0.2)
        img = augmentation_dali.gaussian_blur(img, sigma_range=(0.5, 1.0), p=0.2)
        img = augmentation_dali.gaussian_noise(img, std=0.1, p=0.2)
        
        img, lab = augmentation_dali.random_affine_crop_3d(img, lab, p=0.2, window_size=self.args.training_size, pad_size=self.args.affine_pad_size)
        
        img = fn.crop_mirror_normalize(img, output_layout='CDHW')
        lab = fn.crop_mirror_normalize(lab, output_layout='CDHW')

        return img, lab 
