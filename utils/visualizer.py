import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imsave

import torch


def normalize_tensor(image):
    image += torch.abs(torch.min(image))
    image_max = torch.abs(torch.max(image))
    if image_max > 0:
        image /= image_max
    return image

def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image


def save_image_mask(path, image, mask=None, pred=None):
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    image = (image * 255).astype(np.uint8)
    
    if mask is not None:
        mask = mask.squeeze()
        image[mask > 0] += np.array([0, 63, 0], dtype=np.uint8)
        image = outline(image, mask, color=[0, 255, 0])
    
    if pred is not None:
        pred = pred.squeeze()
        image[pred > 0] += np.array([63, 0, 0], dtype=np.uint8)
        image = outline(image, pred, color=[255, 0, 0])
        
    image = np.clip(image, a_min=0, a_max=255)
    imsave(path, image)

    
def plot_image_mask(image, mask=None, pred=None):
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    image = (image * 255).astype(np.uint8)
    
    if mask is not None:
        mask = mask.squeeze()
        image[mask > 0] += np.array([0, 63, 0], dtype=np.uint8)
        image = outline(image, mask, color=[0, 255, 0])
    
    if pred is not None:
        pred = pred.squeeze()
        # image[pred > 0] += np.array([63, 0, 0], dtype=np.uint8)
        image = outline(image, pred, color=[255, 0, 0])
        
    image = np.clip(image, a_min=0, a_max=255)
    plt.imshow(image) 

    
def plot_samples(samples):
    plt.figure(figsize=(20,10))
    num_samples = len(samples)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.axis("off")
        plot_image_mask(*samples[i])
    plt.show()
        