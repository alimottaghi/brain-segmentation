import numpy as np
import random

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage

import torch
from torchvision.transforms import Compose


class Augmentation(object):

    def __init__(self, pipeline, use_heatmap=True):
        self.pipeline = pipeline
        self.use_heatmap = use_heatmap

    def __call__(self, sample):
        image, mask = sample
        if self.use_heatmap:
            heatmap = HeatmapsOnImage(mask.astype(np.float32), shape=image.shape)
            image_aug, heatmap_aug = self.pipeline(image=image, heatmaps=heatmap)
            mask_aug = heatmap_aug.get_arr()
        else:
            segmap = SegmentationMapsOnImage(mask.astype(np.int32), shape=image.shape)
            image_aug, segmap_aug = self.pipeline(image=image, segmentation_maps=segmap)
            mask_aug = segmap_aug.get_arr()
        sample_aug = image_aug, mask_aug
        return sample_aug


class ToTensor(object):

    def __call__(self, sample):
        image, mask = sample
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        sample_tensor = image_tensor, mask_tensor
        return sample_tensor


class ToImage(object):

    def __call__(self, sample_tensor):
        if len(sample_tensor)==1:
            sample = sample_tensor.detach().cpu().numpy()
            sample = sample.transpose(1, 2, 0)
            return sample
        else:
            sample = []
            for i, image in enumerate(sample_tensor):
                image = image.detach().cpu().numpy()
                image = image.transpose(1, 2, 0)
                sample.append(image)
            return tuple(sample)



def get_transforms(params, mode='week', totensor=False):
    """Get the transformations from JSON file
    
    Nonlinear Transform:
    "elastic": [2, 0.25],
    "identity": 0,
    "cutout": [4, 0.2],
    "solarize": [0, 256]
    
    PIL Transforms:
    "autocontrast": 0,
    "equalize": 0,
    "enhance_brightness": [0.1, 1.8],
    "enhance_color": [0.1, 1.8],
    "enhance_contrast": [0.1, 1.8],
    "enhance_sharpness":[0.1, 1.8],
    "posterize": [4, 4],
    """
    
    if mode=='week':
        augs_dict = params.week_augmentations
    elif mode=='strong':
        augs_dict = params.strong_augmentations
    else:
        raise NotImplemented
    
    iaa_list = []
    num_augs = len(augs_dict)
    for aug in augs_dict:
        if aug == 'num_augmentations':
            num_augs = augs_dict[aug]
        if aug == 'scale':
            scale = tuple(augs_dict[aug])
            iaa_list.append(iaa.Affine(scale=scale))
        if aug == 'rotate':
            rotate = tuple(augs_dict[aug])
            iaa_list.append(iaa.Affine(rotate=rotate))
        if aug == 'translate':
            percent = tuple(augs_dict[aug])
            iaa_list.append(iaa.TranslateX(percent=percent))
            iaa_list.append(iaa.TranslateY(percent=percent))
        if aug == 'shear':
            shear = tuple(augs_dict[aug])
            iaa_list.append(iaa.ShearX(shear=shear))
            iaa_list.append(iaa.ShearX(shear=shear))
        if aug == 'elastic':
            elastic = tuple(augs_dict[aug])
            iaa_list.append(iaa.ElasticTransformation(alpha=(0, elastic[0]), sigma=(0, elastic[1])))
        if aug == 'flip':
            flip = augs_dict[aug]
            iaa_list.append(iaa.Fliplr(flip))
        if aug == 'identity':
            iaa_list.append(iaa.Identity())
        if aug == 'autocontrast':
            iaa_list.append(iaa.pillike.Autocontrast())
        if aug == 'equalize':
            iaa_list.append(iaa.pillike.Equalize())
        if aug == 'enhance_brightness':
            factor = tuple(augs_dict[aug])
            iaa_list.append(iaa.pillike.EnhanceBrightness(factor=factor))
        if aug == 'enhance_color':
            factor = tuple(augs_dict[aug])
            iaa_list.append(iaa.pillike.EnhanceColor(factor=factor))
        if aug == 'enhance_contrast':
            factor = tuple(augs_dict[aug])
            iaa_list.append(iaa.pillike.EnhanceContrast(factor=factor))
        if aug == 'enhance_sharpness':
            factor = tuple(augs_dict[aug])
            iaa_list.append(iaa.pillike.EnhanceSharpness(factor=factor))
        if aug == 'cutout':
            size = tuple(augs_dict[aug])
            iaa_list.append(iaa.Cutout(nb_iterations=(1, size[0]), size=size[1], cval=0))
        if aug == 'posterize':
            nb_bits = tuple(augs_dict[aug])
            iaa_list.append(iaa.color.Posterize(nb_bits=nb_bits))
        if aug == 'solarize':
            threshold = tuple(augs_dict[aug])
            iaa_list.append(iaa.Solarize(0.5, threshold=threshold))
    
    # iaa_augs = iaa.Sequential(iaa.SomeOf(num_augs,iaa_list))
    iaa_augs = iaa.Sequential(iaa_list)
    if totensor:
        transforms = Compose([Augmentation(iaa_augs), ToTensor()])
    else:
        transforms = Augmentation(iaa_augs)
    return transforms
        
    
        
    