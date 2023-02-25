import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali

import pdb

def random_augmentation(probability, augmented, original):
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original

def random_augmentation_twoinputs(probability, augmented_img, original_img, augmented_lab, original_lab):
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented_img + neg_condition * original_img, condition * augmented_lab + neg_condition * original_lab


def gaussian_noise(img, std, mean=0, p=0.5):
    img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, std))) + mean
    return random_augmentation(p, img_noised, img)

def gaussian_blur(img, sigma_range=(0.5, 2.0), p=0.5):
    img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=sigma_range))
    return random_augmentation(p, img_blurred, img)

def brightness(img, additive_range=(-0.1, 0.1), multiply_range=(0.7, 1.3), p=0.5):
    img_adjusted = fn.brightness(img, brightness=fn.random.uniform(range=multiply_range), brightness_shift=fn.random.uniform(range=additive_range))
    return random_augmentation(p, img_adjusted, img)

def contrast(img, contrast_range=(0.65, 1.5), p=0.5):
    mean = fn.reductions.mean(img)
    img = img - mean
    img_adjusted = fn.contrast(img, contrast=fn.random.uniform(range=contrast_range), contrast_center=0)
    img_adjusted = img_adjusted + mean
    img_adjusted = math.clamp(img_adjusted, fn.reductions.min(img), fn.reductions.max(img))
    return random_augmentation(p, img_adjusted, img)

def mirror(img, lab, axis='h', p=0.5):
    '''
    Args:
        img: axis index is 'dhw'
        Add char 'd' or 'h' or 'w' into axis for flipping
    '''
    kwargs = {}
    if 'd' in axis:
        kwargs['depthwise'] = fn.random.coin_flip(probability=p)
    if 'h' in axis:
        kwargs['horizontal'] = fn.random.coin_flip(probability=p)
    if 'w' in axis:
        kwargs['vertical'] = fn.random.coin_flip(probability=p)
    return fn.flip(img, **kwargs), fn.flip(lab, **kwargs)

def random_affine_2d(img, lab, scale=0.3, rotate=45, translate=20, shear=0.3, window_size=[256, 256], p=0.5):

    center = [i//2 for i in window_size]

    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 2
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 2
    if isinstance(shear, float) or isinstance(shear, int):
        shear = [shear] * 2

    scale_x = fn.random.uniform(range=(1-scale[0], 1+scale[0]))
    scale_y = fn.random.uniform(range=(1-scale[1], 1+scale[1]))
    scale = fn.stack(scale_x, scale_y)
    mt = fn.transforms.scale(scale=scale, center=center)
    
    shear_x = fn.random.uniform(range=(-shear[0], shear[0]))
    shear_y = fn.random.uniform(range=(-shear[1], shear[1]))
    shear = fn.stack(shear_x, shear_y)
    mt = fn.transforms.shear(mt, shear=shear, center=center)
    
    angle = fn.random.uniform(range=(-rotate, rotate))
    mt = fn.transforms.rotation(mt, angle=angle, center=center)
    
    translate_x = fn.random.uniform(range=(-translate[0], translate[0]))
    translate_y = fn.random.uniform(range=(-translate[1], translate[1]))
    translate = fn.stack(translate_x, translate_y)
    
    mt = fn.transforms.translation(mt, offset=translate)
    
    transformed_img = fn.warp_affine(img, matrix=mt, fill_value=0, inverse_map=False, interp_type=types.DALIInterpType.INTERP_LINEAR)
    transformed_lab = fn.warp_affine(lab, matrix=mt, fill_value=0, inverse_map=False, interp_type=types.DALIInterpType.INTERP_NN)
 
    return random_augmentation_twoinputs(p, transformed_img, img, transformed_lab, lab)


def random_affine_3d(img, lab, scale=0.3, rotate=45, translate=20, shear=0.2, window_size=[128, 128, 128], p=0.5):

    center = [i//2 for i in window_size]
    
    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 3
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 3
    if isinstance(shear, float) or isinstance(shear, int):
        shear = [shear] * 3
    if isinstance(rotate, float) or isinstance(rotate, int):
        rotate = [rotate] * 3

    scale_x = fn.random.uniform(range=(1-scale[0], 1/(1-scale[0])))
    scale_y = fn.random.uniform(range=(1-scale[1], 1/(1-scale[1])))
    scale_z = fn.random.uniform(range=(1-scale[2], 1/(1-scale[2])))
    scale = fn.stack(scale_x, scale_y, scale_z)
    mt = fn.transforms.scale(scale=scale, center=center)
    
    shear_xy = fn.random.uniform(range=(-shear[0], shear[0]))
    shear_xz = fn.random.uniform(range=(-shear[0], shear[0]))
    shear_yx = fn.random.uniform(range=(-shear[1], shear[1]))
    shear_yz = fn.random.uniform(range=(-shear[1], shear[1]))
    shear_zx = fn.random.uniform(range=(-shear[2], shear[2]))
    shear_zy = fn.random.uniform(range=(-shear[2], shear[2]))

    shear_x = fn.stack(shear_xy, shear_xz)
    shear_y = fn.stack(shear_yx, shear_yz)
    shear_z = fn.stack(shear_zx, shear_zy)
    shear = fn.stack(shear_x, shear_y, shear_z)
    mt = fn.transforms.shear(mt, shear=shear, center=center)
    
    angle_x = fn.random.uniform(range=(-rotate[0], rotate[0]))
    angle_y = fn.random.uniform(range=(-rotate[1], rotate[1]))
    angle_z = fn.random.uniform(range=(-rotate[2], rotate[2]))

    mt = fn.transforms.rotation(mt, angle=angle_x, axis=(1,0,0), center=center)
    mt = fn.transforms.rotation(mt, angle=angle_y, axis=(0,1,0), center=center)
    mt = fn.transforms.rotation(mt, angle=angle_z, axis=(0,0,1), center=center)

    translate_x = fn.random.uniform(range=(-translate[0], translate[0]))
    translate_y = fn.random.uniform(range=(-translate[1], translate[1]))
    translate_z = fn.random.uniform(range=(-translate[2], translate[2]))
    translate = fn.stack(translate_x, translate_y, translate_z)
    mt = fn.transforms.translation(mt, offset=translate)
    
    #mt = random_augmentation(p, mt, fn.transforms.scale(scale=(1,1,1), center=center))
    
    transformed_img = fn.warp_affine(img, matrix=mt, fill_value=0, inverse_map=False, interp_type=types.DALIInterpType.INTERP_LINEAR)
    transformed_lab = fn.warp_affine(lab, matrix=mt, fill_value=0, inverse_map=False, interp_type=types.DALIInterpType.INTERP_NN)
    
    return random_augmentation_twoinputs(p, transformed_img, img, transformed_lab, lab)
    

def random_affine_crop_3d(img, lab, scale=0.3, rotate=45, translate=20, shear=0.2, 
                        window_size=[128, 128, 128], pad_size=[30,30,30], p=0.5):
    # Same with the above random_affine_3d, but with crop trick to save computation & memory

    pad_center = [(i+j)//2 for i, j in zip(window_size, pad_size)]
    pad_size = [i+j for i, j in zip(window_size, pad_size)]
    
    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 3
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 3
    if isinstance(shear, float) or isinstance(shear, int):
        shear = [shear] * 3
    if isinstance(rotate, float) or isinstance(rotate, int):
        rotate = [rotate] * 3

    scale_x = fn.random.uniform(range=(1-scale[0], 1/(1-scale[0])))
    scale_y = fn.random.uniform(range=(1-scale[1], 1/(1-scale[1])))
    scale_z = fn.random.uniform(range=(1-scale[2], 1/(1-scale[2])))
    scale = fn.stack(scale_x, scale_y, scale_z)
    mt = fn.transforms.scale(scale=scale, center=pad_center)
    
    shear_xy = fn.random.uniform(range=(-shear[0], shear[0]))
    shear_xz = fn.random.uniform(range=(-shear[0], shear[0]))
    shear_yx = fn.random.uniform(range=(-shear[1], shear[1]))
    shear_yz = fn.random.uniform(range=(-shear[1], shear[1]))
    shear_zx = fn.random.uniform(range=(-shear[2], shear[2]))
    shear_zy = fn.random.uniform(range=(-shear[2], shear[2]))

    shear_x = fn.stack(shear_xy, shear_xz)
    shear_y = fn.stack(shear_yx, shear_yz)
    shear_z = fn.stack(shear_zx, shear_zy)
    shear = fn.stack(shear_x, shear_y, shear_z)
    mt = fn.transforms.shear(mt, shear=shear, center=pad_center)
    
    angle_x = fn.random.uniform(range=(-rotate[0], rotate[0]))
    angle_y = fn.random.uniform(range=(-rotate[1], rotate[1]))
    angle_z = fn.random.uniform(range=(-rotate[2], rotate[2]))

    mt = fn.transforms.rotation(mt, angle=angle_x, axis=(1,0,0), center=pad_center)
    mt = fn.transforms.rotation(mt, angle=angle_y, axis=(0,1,0), center=pad_center)
    mt = fn.transforms.rotation(mt, angle=angle_z, axis=(0,0,1), center=pad_center)

    translate_x = fn.random.uniform(range=(-translate[0], translate[0]))
    translate_y = fn.random.uniform(range=(-translate[1], translate[1]))
    translate_z = fn.random.uniform(range=(-translate[2], translate[2]))
    translate = fn.stack(translate_x, translate_y, translate_z)
    mt = fn.transforms.translation(mt, offset=translate)
    
    #mt = random_augmentation(p, mt, fn.transforms.scale(scale=(1,1,1), center=center))
    # Random crop to pad size
    crop_trick_x, crop_trick_y, crop_trick_z = fn.random.uniform(range=(0, 1)), fn.random.uniform(range=(0, 1)), fn.random.uniform(range=(0, 1))
    cropped_img = fn.crop(img, crop=pad_size, crop_pos_x=crop_trick_x, crop_pos_y=crop_trick_y, crop_pos_z=crop_trick_z)
    cropped_lab = fn.crop(lab, crop=pad_size, crop_pos_x=crop_trick_x, crop_pos_y=crop_trick_y, crop_pos_z=crop_trick_z)
    
    # Affine transformation
    transformed_img = fn.warp_affine(cropped_img, matrix=mt, fill_value=0, inverse_map=False, interp_type=types.DALIInterpType.INTERP_LINEAR)
    transformed_lab = fn.warp_affine(cropped_lab, matrix=mt, fill_value=0, inverse_map=False, interp_type=types.DALIInterpType.INTERP_NN)
    # Center crop to training window size
    transformed_img = fn.crop(transformed_img, crop=window_size, crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5)
    transformed_lab = fn.crop(transformed_lab, crop=window_size, crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5)

    # For probability p, use origin random cropped window size sample
    origin_x, origin_y, origin_z = fn.random.uniform(range=(0, 1)), fn.random.uniform(range=(0, 1)), fn.random.uniform(range=(0, 1))
    origin_img = fn.crop(img, crop=window_size, crop_pos_x=origin_x, crop_pos_y=origin_y, crop_pos_z=origin_z)
    origin_lab = fn.crop(lab, crop=window_size, crop_pos_x=origin_x, crop_pos_y=origin_y, crop_pos_z=origin_z)
    
    return random_augmentation_twoinputs(p, transformed_img, origin_img, transformed_lab, origin_lab)
    






