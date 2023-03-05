import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, ITKReDirection
import os
import random
import yaml
import copy
import pdb

from matplotlib import image

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    assert round(imImage.GetSpacing()[0], 2) == round(imLabel.GetSpacing()[0], 2)
    assert round(imImage.GetSpacing()[1], 2) == round(imLabel.GetSpacing()[1], 2)
    assert round(imImage.GetSpacing()[2], 2) == round(imLabel.GetSpacing()[2], 2)

    assert imImage.GetSize() == imLabel.GetSize()


    imLabel.CopyInformation(imImage)
    
    imImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imLabel.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()
    

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
       

    re_img_yz = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_yz = ResampleLabelToRef(imLabel, re_img_yz, interp=sitk.sitkNearestNeighbor)
    
    re_img_xyz = ResampleXYZAxis(re_img_yz, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_yz, re_img_xyz, interp=sitk.sitkNearestNeighbor)


    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[30, 30, 30]) # z, y, x

    sitk.WriteImage(cropped_img, '%s/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz'%(save_path, name))


if __name__ == '__main__':


    src_path = '/filer/tmp1/yg397/dataset/amos/amos22/'
    ct_tgt_path = '/filer/tmp1/yg397/dataset/amos/cbim/amos_ct_3d/'
    mr_tgt_path = '/filer/tmp1/yg397/dataset/amos/cbim/amos_mr_3d/'


    print('Start processing training set')
    ct_name_list = []
    mr_name_list = []
    for name in os.listdir(f"{src_path}imagesTr/"):
        if not name.endswith('nii.gz'):
            continue
        print(name)
        idx = name.split('.')[0]
        idx = int(idx.split('_')[1])
        if idx < 500:
            ct_name_list.append(idx)
        else:
            mr_name_list.append(idx)
        

    if not os.path.exists(ct_tgt_path+'list'):
        os.mkdir('%slist'%(ct_tgt_path))
    with open("%slist/dataset.yaml"%ct_tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(ct_name_list, f)
    
    if not os.path.exists(mr_tgt_path+'list'):
        os.mkdir('%slist'%(mr_tgt_path))
    with open("%slist/dataset.yaml"%mr_tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(mr_name_list, f)

    os.chdir(src_path)
    
    for name in ct_name_list:
        img = sitk.ReadImage(src_path+f"imagesTr/amos_{name:04d}.nii.gz")
        lab = sitk.ReadImage(src_path+f"labelsTr/amos_{name:04d}.nii.gz")

        ResampleImage(img, lab, ct_tgt_path, name, (0.68825, 0.68825, 2.0))
        print(name, 'done')
    
    for name in mr_name_list:
        img = sitk.ReadImage(src_path+f"imagesTr/amos_{name:04d}.nii.gz")
        lab = sitk.ReadImage(src_path+f"labelsTr/amos_{name:04d}.nii.gz")

        ResampleImage(img, lab, mr_tgt_path, name, (1.1875, 1.1875, 2.0))
        print(name, 'done')
    
    print('Start processing validation set') 
    ct_name_list = []
    mr_name_list = []
    for name in os.listdir(f"{src_path}imagesVa/"):
        if not name.endswith('nii.gz'):
            continue
        print(name)
        idx = name.split('.')[0]
        idx = int(idx.split('_')[1])
        if idx < 500:
            ct_name_list.append(idx)
        else:
            mr_name_list.append(idx)
    
    for name in ct_name_list:
        img = sitk.ReadImage(src_path+f"imagesVa/amos_{name:04d}.nii.gz")
        lab = sitk.ReadImage(src_path+f"labelsVa/amos_{name:04d}.nii.gz")

        ResampleImage(img, lab, ct_tgt_path, name, (0.68825, 0.68825, 2.0))
        print(name, 'done')
    
    for name in mr_name_list:
        img = sitk.ReadImage(src_path+f"imagesVa/amos_{name:04d}.nii.gz")
        lab = sitk.ReadImage(src_path+f"labelsVa/amos_{name:04d}.nii.gz")

        ResampleImage(img, lab, mr_tgt_path, name, (1.1875, 1.1875, 2.0))
        print(name, 'done')
    
