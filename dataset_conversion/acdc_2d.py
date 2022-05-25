import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef
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
    
    
    re_img = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab = ResampleLabelToRef(imLabel, re_img, interp=sitk.sitkNearestNeighbor)


    sitk.WriteImage(re_img, '%s/%s_%d.nii.gz'%(save_path, patient_name, count))
    sitk.WriteImage(re_lab, '%s/%s_%d_gt.nii.gz'%(save_path, patient_name, count))



if __name__ == '__main__':


    src_path = '/research/cbim/medical/medical-share/public/ACDC/raw/training/'
    tgt_path = '/research/cbim/medical/yg397/tgt_dir/'




    patient_list = list(range(1, 101))

    name_list = []
    for idx in patient_list:
        name_list.append('patient%.3d'%idx)

    if not os.path.exists(tgt_path+'list'):
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
                print(name, '%d'%count, 'done')

        os.chdir('..')


