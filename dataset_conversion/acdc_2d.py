import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleFullImageToRef
import os
import random
import yaml
import pdb

def ResampleCMRImage(imImage, imLabel, save_path, patient_name, count, target_spacing=(1., 1., 1.)):

    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()


    npimg = sitk.GetArrayFromImage(imImage)
    nplab = sitk.GetArrayFromImage(imLabel)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
    
    tmp_img = npimg
    tmp_lab = nplab

    tmp_itkimg = sitk.GetImageFromArray(tmp_img)
    tmp_itkimg.SetSpacing(spacing[0:3])
    tmp_itkimg.SetOrigin(origin[0:3])
    tmp_itkimg.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    tmp_itklab = sitk.GetImageFromArray(tmp_lab)
    tmp_itklab.SetSpacing(spacing[0:3])
    tmp_itklab.SetOrigin(origin[0:3])
    tmp_itklab.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    
    re_img = ResampleXYZAxis(tmp_itkimg, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab = ResampleFullImageToRef(tmp_itklab, re_img)


    sitk.WriteImage(re_img, '%s/%s_%d.nii.gz'%(save_path, patient_name, count))
    sitk.WriteImage(re_lab, '%s/%s_%d_gt.nii.gz'%(save_path, patient_name, count))



if __name__ == '__main__':


    src_path = '/research/cbim/medical/medical-share/public/ACDC/raw/training/'
    tgt_path = '/research/cbim/medical/yg397/ACDC_2d/'



    # This is to align train/val/test split with TransUNet SwinUNet and etc.

    patient_list = list(range(1, 101))

    #val_list = [89, 90, 91, 93, 94, 96, 97, 98, 99, 100]
    #test_list = [2, 3, 8, 9, 12, 14, 17, 24, 42, 48, 49, 53, 55, 64, 67, 79, 81, 88, 92, 95]
    #train_list = list(set(patient_list) - set(val_list) - set(test_list))

    # If don't want to align with them, just use following code to random split
    '''
    patient_list = list(range(1, 101))
    random.seed(0)
    random.shuffle(patient_list)

    train_list = patient_list[:70]
    val_list = patient_list[70:80]
    test_list = patient_list[80:]
    '''
    
    name_list = []
    for idx in patient_list:
        name_list.append('patient%.3d'%idx)




    os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)
    for name in os.listdir('.'):
        os.chdir(name)
        
        count = 0
        for i in os.listdir('.'):
            if 'gt' in i:
                tmp = i.split('_')
                img_name = tmp[0] + '_' + tmp[1]
                patient_name = tmp[0]

                img = sitk.ReadImage('%s.nii.gz'%img_name)
                lab = sitk.ReadImage('%s_gt.nii.gz'%img_name)

                ResampleCMRImage(img, lab, tgt_path, patient_name, count, (1.5625, 1.5625))
                count += 1
                print(name, 'done')

        os.chdir('..')


