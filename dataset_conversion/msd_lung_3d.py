import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground
from skimage import measure
from skimage import morphology
import os
import random
import yaml
import copy
import pdb


def CropBody(itkImg, itkLab):
    npimg = sitk.GetArrayFromImage(itkImg)
    nplab = sitk.GetArrayFromImage(itkLab)
    
    mask = (npimg > 500).astype(np.uint8)
    #mask = morphology.remove_small_objects(measure.label(mask), min_size=50)
    
    labeled_mask = measure.label(mask, connectivity=3)
    regions = measure.regionprops(labeled_mask)
    print('num of regions:', len(regions))
    
    largest_idx = -1
    largest_size = 0
    for i in range(len(regions)):
        if regions[i].area > largest_size:
            largest_size = regions[i].area
            largest_idx = i
   
    zz, yy, xx = npimg.shape
    z_min, y_min, x_min, z_max, y_max, x_max = regions[largest_idx].bbox


    x_min = max(0, x_min - 60)
    x_max = min(xx, x_max + 60)
    y_max = min(yy, y_max+30)
    cropped_img = npimg[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_lab = nplab[z_min:z_max, y_min:y_max, x_min:x_max]

    print(x_max-x_min, y_max-y_min, z_max-z_min)

    cropped_itk_img = sitk.GetImageFromArray(cropped_img)
    cropped_itk_lab = sitk.GetImageFromArray(cropped_lab)

    cropped_itk_img.SetSpacing(itkImg.GetSpacing())
    cropped_itk_img.SetDirection(itkImg.GetDirection())
    cropped_itk_lab.SetSpacing(itkImg.GetSpacing())
    cropped_itk_lab.SetDirection(itkImg.GetDirection())

    return cropped_itk_img, cropped_itk_lab



def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    assert round(imImage.GetSpacing()[0], 2) == round(imLabel.GetSpacing()[0], 2)
    assert round(imImage.GetSpacing()[1], 2) == round(imLabel.GetSpacing()[1], 2)
    assert round(imImage.GetSpacing()[2], 2) == round(imLabel.GetSpacing()[2], 2)

    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()
    
    imLabel.CopyInformation(imImage)

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))


    re_img_yz = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_yz = ResampleLabelToRef(imLabel, re_img_yz, interp=sitk.sitkNearestNeighbor)
    
    re_img_xyz = ResampleXYZAxis(re_img_yz, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_yz, re_img_xyz, interp=sitk.sitkNearestNeighbor)


    
    cropped_img, cropped_lab = CropBody(re_img_xyz, re_lab_xyz)

    sitk.WriteImage(cropped_img, '%s/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz'%(save_path, name))


if __name__ == '__main__':


    src_path = '/research/cbim/medical/yg397/MSD_Lung/Task06_Lung/'
    tgt_path = '/research/cbim/medical/yg397/MSD_Lung/msd_lung_3d/'

    
    name_list = []
    for i in os.listdir(src_path + 'imagesTr'):
        if i.startswith('.'):
            continue
        name_list.append(i.split('.')[0])

    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)

    for name in name_list:
        img = sitk.ReadImage(src_path+f"imagesTr/{name}.nii.gz")
        lab = sitk.ReadImage(src_path+f"labelsTr/{name}.nii.gz")

        ResampleImage(img, lab, tgt_path, name, (0.78515625, 0.78515625, 1.244979977607727))
        print(name, 'done')


