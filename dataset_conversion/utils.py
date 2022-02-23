import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops

import os

def ResampleXYZAxis(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

    imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())

    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage

def ResampleFullImageToRef(imImage, imRef, interp=sitk.sitkNearestNeighbor):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    imRefImage = sitk.Image(imRef.GetSize(), imImage.GetPixelIDValue())
    imRefImage.SetSpacing(imRef.GetSpacing())
    imRefImage.SetOrigin(imRef.GetOrigin())
    imRefImage.SetDirection(imRef.GetDirection())


    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage


def CropForeground(imImage, imLabel, context_size=[10, 30, 30]):
    
    npImg = sitk.GetArrayFromImage(imImage)
    npLab = sitk.GetArrayFromImage(imLabel)

    mask = (npLab>0).astype(np.uint8) # foreground mask

    regions = regionprops(mask)
    assert len(regions) == 1

    zz, xx, yy = npImg.shape

    z, x, y = regions[0].centroid

    z_min, x_min, y_min, z_max, x_max, y_max = regions[0].bbox
    print('forground size:', z_max-z_min, x_max-x_min, y_max-y_min)

    z, x, y = int(z), int(x), int(y)

    z_min = max(0, z_min-context_size[0])
    z_max = min(zz, z_max+context_size[0])
    x_min = max(0, x_min-context_size[1])
    x_max = min(xx, x_max+context_size[1])
    y_min = max(0, y_min-context_size[2])
    y_max = min(yy, y_max+context_size[2])

    img = npImg[z_min:z_max, x_min:x_max, y_min:y_max]
    lab = npLab[z_min:z_max, x_min:x_max, y_min:y_max]

    croppedImage = sitk.GetImageFromArray(img)
    croppedLabel = sitk.GetImageFromArray(lab)


    croppedImage.SetSpacing(imImage.GetSpacing())
    croppedLabel.SetSpacing(imImage.GetSpacing())
    
    croppedImage.SetDirection(imImage.GetDirection())
    croppedLabel.SetDirection(imImage.GetDirection())

    return croppedImage, croppedLabel





