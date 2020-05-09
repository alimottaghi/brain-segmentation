import os
import random
import numpy as np

from torch.utils.data import Dataset
from tifffile import imread, imsave


class BrainSegmentationDataset(Dataset):

    def __init__(self, root_dir, subset="train", transform=None, params=None):
        self.subset = subset
        self.transform = transform
        self.image_files, self.mask_files = list_images(root_dir, subset, params)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        image = imread(image_path)
        mask = imread(mask_path)
        sample = (image, mask)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    

    
def list_images(root_dir, subset, params):
    assert subset in ["train", "validation", "labeled", "unlabeled"]
    
    all_image_files = {}
    all_mask_files = {}
    for (dirpath, dirnames, filenames) in os.walk(root_dir):
        image_slices = []
        mask_slices = []
        for filename in sorted(filter(lambda f: ".tif" in f, filenames),
                               key=lambda x: int(x.split(".")[-2].split("_")[4])):
            filepath = os.path.join(dirpath, filename)
            if "mask" in filename:
                mask_slices.append(filepath)
            else:
                image_slices.append(filepath)
        if len(image_slices) > 0:
            patient_id = dirpath.split("/")[-1]
            all_image_files[patient_id] = image_slices
            all_mask_files[patient_id] = mask_slices
    all_patients = sorted(all_image_files)
        
    random.seed(params.seed)
    validation_patients = random.sample(all_patients, k=params.num_validation_patients)
    train_patients = sorted(list(set(all_patients).difference(validation_patients)))
    labeled_patients = random.sample(train_patients, k=params.num_labeled_patients)
    unlabeled_patients = sorted(list(set(train_patients).difference(labeled_patients)))
        
    if subset == "train":
        patients = train_patients
    elif subset == "validation":
        patients = validation_patients
    elif subset == "unlabeled":
        patients = unlabeled_patients
    else:
        patients = labeled_patients
        
    subset_image_files = []
    subset_mask_files = []
    for patient_id in patients:
        subset_image_files.extend(all_image_files[patient_id])
        subset_mask_files.extend(all_mask_files[patient_id])
    return subset_image_files, subset_mask_files
        

