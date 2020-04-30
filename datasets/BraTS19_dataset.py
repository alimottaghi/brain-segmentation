"""
Pytorch framework for Semi-supervised learning in Medical Image Analysis

BraTS18

Author(s): Shuai Chen
PhD student in Erasmus MC, Rotterdam, the Netherlands
Biomedical Imaging Group Rotterdam

If you have any questions or suggestions about the code, feel free to contact me:
Email: chenscool@gmail.com

Date: 22 Jan 2019
"""
import sys
sys.path.append("/pasteur/u/ozt/medical-image-analysis-fewer-labels/MASSL-segmentation-framework")
import numpy as np
import module.common_module as cm
from glob import glob
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms, utils
import module.transform as trans
import torch

# Build Dataset Class
class BraTSDataset(Dataset):
    """Segmentation dataset

    Image: T1, Flair [Channel, Z, H, W]
    Labels: 0 - background, 1 - White matter lesion
    """

    def __init__(self, img_list, mask_list, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        # assert len(self.img_list) == len(self.mask_list)
        return len(self.img_list)

    def __getitem__(self, idx):
        image = np.load(self.img_list[idx])
        mask = np.load(self.mask_list[idx])
        image_slice = image[0:3,80,:,:]
        sample = (np.moveaxis(image_slice,0,-1),np.reshape(mask[80,:,:],(200,200,1)))

        if self.transform:
            sample = self.transform(sample)

        return sample

def BraTS19data(data_seed,transform,transform2):

    # Set random seed
    data_seed = data_seed
    np.random.seed(data_seed)

    # Create image list
    HGG_imgList = sorted(glob('/pasteur/u/ozt/brats/MICCAI_BraTS_2019_Data_Training/HGG/*_img.npy'))
    HGG_maskList = sorted(glob('/pasteur/u/ozt/brats/MICCAI_BraTS_2019_Data_Training/HGG/*_mask.npy'))
    HGG_imgList_val = sorted(glob('/pasteur/u/ozt/brats/MICCAI_BraTS_2019_Data_Validation/*_img.npy'))
    HGG_maskList_val = sorted(glob('/pasteur/u/ozt/brats/MICCAI_BraTS_2019_Data_Validation/*_img.npy'))
    # Random selection for training, validation and testing
    HGG_list = [list(pair) for pair in zip(HGG_imgList, HGG_maskList)]
    HGG_list_val = [list(pair) for pair in zip(HGG_imgList_val, HGG_maskList_val)]
    print(len(HGG_list))
    print(len(HGG_list_val))
    np.random.shuffle(HGG_list)
    np.random.shuffle(HGG_list_val)
   
    train_labeled_img_list = []
    train_labeled_mask_list = []
    train_unlabeled_img_list = []
    train_unlabeled_mask_list = []
    val_labeled_img_list = []
    val_labeled_mask_list = []
    val_unlabeled_img_list = []
    val_unlabeled_mask_list = []
    test_img_list = []
    test_mask_list = []


    train_labeled_img_list, train_labeled_mask_list = map(list, zip(*(HGG_list[0:  150])))
    train_unlabeled_img_list, train_unlabeled_mask_list  = map(list, zip(*(HGG_list_val[0: 125])))
    val_labeled_img_list, val_labeled_mask_list = map(list, zip(*(HGG_list[150:200])))
    val_unlabeled_img_list,val_unlabeled_mask_list = map(list, zip(*(HGG_list[150:200])))
    test_img_list, test_mask_list = map(list, zip(*(HGG_list[200:250])))

    # Reset random seed
    seed = random.randint(1, 9999999)
    np.random.seed(seed + 1)




    # Iterating through the dataset
    trainLabeledDataset = BraTSDataset(train_labeled_img_list, train_labeled_mask_list,
                                     transform=transform
                                     )
                                     

    trainUnlabeledDataset = BraTSDataset(train_unlabeled_img_list, train_unlabeled_mask_list,
                                       transform=transform2
                                       )
                                       

    valLabeledDataset = BraTSDataset(val_labeled_img_list, val_labeled_mask_list,
                                   transform=transform2
                                   )

    valUnlabeledDataset = BraTSDataset(val_unlabeled_img_list, val_unlabeled_mask_list,
                                     transform=transform
                                     )

    testDataset = BraTSDataset(test_img_list, test_mask_list,
                             transform=transforms.Compose([
                                 trans.ToTensor()
                             ])
                             )

    # device_type = 'cpu'
    device_type = 'cuda'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    dataset_sizes = {'trainLabeled': len(trainLabeledDataset), 'trainUnlabeled': len(trainUnlabeledDataset),
                     'val_labeled': len(valLabeledDataset), 'val_unlabeled': len(valUnlabeledDataset),
                     'test': len(testDataset)}

    modelDataLoader = {'trainLabeled': DataLoader(trainLabeledDataset, batch_size=1, shuffle=True, num_workers=4),
                       'trainUnlabeled': DataLoader(trainUnlabeledDataset, batch_size=1, shuffle=True, num_workers=4),
                       'val_labeled': DataLoader(valLabeledDataset, batch_size=1, shuffle=True, num_workers=4),
                       'val_unlabeled': DataLoader(valUnlabeledDataset, batch_size=1, shuffle=True, num_workers=4),
                       'test': DataLoader(testDataset, batch_size=1, shuffle=True, num_workers=4)}

    return trainLabeledDataset,trainUnlabeledDataset,valLabeledDataset
    #return device, dataset_sizes, modelDataLoader
