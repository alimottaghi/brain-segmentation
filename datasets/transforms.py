import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch


class Augmentation(object):

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, sample):
        image, mask = sample
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
        