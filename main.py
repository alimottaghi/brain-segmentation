import os
import json
import argparse
import logging
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.nn as nn

from datasets.brain_dataset import BrainSegmentationDataset
from datasets.pascal_voc_dataset import PascalSegmentationDataset
from datasets.bdd_dataset import BDDSegmentationDataset, median_frequency_balance
from datasets.cityscapes import CityscapesSegmentationDataset
from datasets.transforms import ToTensor, ToImage, get_transforms
from models.unet import UNet
from metrics.dice_loss import DiceLoss, dice_score
from metrics.metrics import compute_iou
from train import train_eval
from evaluate import generate_outputs
from utils.params import Params, synthesize_results
from utils.logger import set_logger
from utils.visualizer import plot_samples


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=None, help="Directory containing the dataset")
parser.add_argument('--exp_dir', default=None, help='Directory to save the results')
parser.add_argument('--base_file', default='base_params.json', help="Path of base_params.json file")
parser.add_argument('--alg', default='supervised', help="Algorithm for the training (supervised or semi-supervised)")
parser.add_argument('--run', default='run0', help="Run suffix")
parser.add_argument('--restore', default=True, help="Restore the previous checkpoint (if exists) or not")
parser.add_argument('--search_params', type=json.loads, default='{"num_labeled_patients": [10, 50, 90]}', help="Dictionary for hyperparameters to tune (in string format)")


def lunch_training_job(algorithm, model_dir, data_dir, params):
    """Launch training of the model with a set of hyperparameters
    Args:
        algorithm: (string) algorithm for training e.g. 'supervised', 'semi-supervised'
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    
    os.makedirs(model_dir, exist_ok=True)
    params.model_dir = model_dir
    params.data_dir = data_dir
    
    set_logger(model_dir)
    
    if params.dataset == 'brats' or params.dataset == 'brain-segmentation':
        week_transforms = get_transforms(params, mode='week')
        strong_transforms = get_transforms(params, mode='strong')
    else:
        week_transforms = get_transforms(params, mode='week',totensor=True)
        strong_transforms = get_transforms(params, mode='strong',totensor=True)
    totensor = ToTensor()
    toimage = ToImage()
    train_transforms = Compose([week_transforms, totensor])
    params.week_transforms = week_transforms
    params.strong_transforms = strong_transforms
    
    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if params.dataset == 'brats' or params.dataset == 'brain-segmentation':
        if params.model == 'deeplab':
            num_classes = 2
        else:
            num_classes = 1
        if params.dataset == 'brats':
            data_dir = "/pasteur/u/ozt/brats/MICCAI_BraTS_2019_Data_Training/HGG/"
        labeled_dataset = BrainSegmentationDataset(data_dir, subset="labeled", transform=train_transforms, params=params)
        unlabeled_dataset = BrainSegmentationDataset(data_dir, subset="unlabeled", transform=train_transforms, params=params)
        val_dataset = BrainSegmentationDataset(data_dir, subset="validation", transform=totensor, params=params)
    elif params.dataset == 'pascal':
        num_classes = 21
        data_dir = "/pasteur/u/ozt/VOC2012/"
        sbd_path = "/pasteur/u/ozt/benchmark_RELEASE/"
        labeled_dataset = PascalSegmentationDataset(data_dir, split='train_aug',subset='labeled',sbd_path = sbd_path, augmentations=week_transforms,params=params)
        unlabeled_dataset = PascalSegmentationDataset(data_dir, split='train_aug',subset='unlabeled',sbd_path = sbd_path, augmentations=totensor,params=params)
        val_dataset = PascalSegmentationDataset(data_dir, split='train_aug_val',subset='train',sbd_path = sbd_path, augmentations=totensor,params=params)
        
    elif params.dataset == 'cityscapes':
        num_classes = 19
        data_dir = "/pasteur/u/ozt/cityscape/"
        train_dataset =  CityscapesSegmentationDataset(data_dir, split='train', augmentations=week_transforms)
        val_dataset =  CityscapesSegmentationDataset('/pasteur/u/ozt/cityscape/', split='val',augmentations=totensor)
        
        partial_size = int(params.labeled_ratio * train_dataset.__len__())
        train_ids = np.arange(train_dataset.__len__())
        np.random.shuffle(train_ids)
        labeled_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        unlabeled_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        
    if params.dataset == 'cityscapes': 
        labeled_loader = data.DataLoader(train_dataset,
                    batch_size=params.batch_size, sampler=labeled_sampler, drop_last = True )
        unlabeled_loader = data.DataLoader(train_dataset,
                    batch_size=params.batch_size, sampler=unlabeled_sampler, drop_last = True )
        val_loader = DataLoader(val_dataset, batch_size=params.batch_size, drop_last= True )
    else:
        labeled_loader = DataLoader(labeled_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=params.batch_ratio*params.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True, drop_last=False)
        
    if params.model == 'unet':
        model = UNet(in_channels=3, out_channels=num_classes)        
    elif params.model == 'deeplab':
        if params.backbone == 'xception':
            model = DeepLab(Xception(output_stride=16), num_classes=num_classes)
        elif params.backbone == 'resnet': 
            model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
            
    model.to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    if params.loss == 'dice':
            loss_fn = DiceLoss()
    elif params.loss == 'crossentropy':
            loss_fn = nn.CrossEntropyLoss(ignore_index=250)
    if params.metric == 'dice':
        metrics = {'dice score': dice_score}
    elif params.metric == 'iou': 
        metrics = {'dice score':compute_iou}
    
    train_eval(algorithm, model, optimizer, loss_fn, labeled_loader, unlabeled_loader, val_loader, metrics, params)
        
    output_list = generate_outputs(model, val_loader, params, save=True)

    
if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.data_dir is None:
        if 'sailhome' in os.getcwd():
            data_dir = '/pasteur/data/lgg-mri-segmentation/processed'
        elif 'users' in os.getcwd():
            data_dir = '/home/groups/syyeung/lgg-mri-segmentation/processed'
        elif '/home/mottaghi' in os.getcwd():
            data_dir = '/home/mottaghi/data/lgg-mri-segmentation/processed'
        elif 'ozt' in os.getcwd():
            data_dir = '/pasteur/u/ozt/lgg-mri-segmentation'
        else:
            raise NotImplemented
    
    if args.exp_dir is None:
        if 'sailhome' in os.getcwd():
            exp_dir = '/pasteur/u/mottaghi/brain-segmentation/experiments'
            # exp_dir = '/pasteur/results/brain-segmentation/experiments'
        elif 'users' in os.getcwd():
            exp_dir = '/scratch/users/mottaghi/brain-segmentation/experiments'
            # exp_dir = '/scratch/groups/syyeung/brain-segmentation/experiments'
        elif '/home/mottaghi' in os.getcwd():
            exp_dir = '/home/mottaghi/experiments/'
        elif 'ozt' in os.getcwd():
            exp_dir = '/pasteur/u/ozt/experiments/'
        else:
            raise NotImplemented
    
    if args.search_params:
        for param_name in args.search_params:
            for val in args.search_params[param_name]:
                print('--- Hyperparameter ' + param_name + ' : ' + str(val))
                model_dir = os.path.join(exp_dir, param_name, args.alg + '_' + str(val) + '_' + args.run)
                os.makedirs(model_dir, exist_ok=True)
                params = Params(args.base_file)
                params.restore = args.restore
                setattr(params, param_name, val)
                params_path = os.path.join(model_dir, "params.json")
                params.save(params_path)
                lunch_training_job(args.alg, model_dir, data_dir, params)
    else:
        model_dir = os.path.join(exp_dir, args.alg + '_' + args.run)
        os.makedirs(model_dir, exist_ok=True)
        params = Params(args.base_file)
        params.restore = args.restore
        params_path = os.path.join(model_dir, "params.json")
        params.save(params_path)
        lunch_training_job(args.alg, model_dir, data_dir, params)
    
    synthesize_results(exp_dir)

    