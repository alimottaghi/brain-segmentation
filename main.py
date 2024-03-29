import os
import json
import argparse
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from datasets.brain_dataset import BrainSegmentationDataset
from datasets.transforms import ToTensor, ToImage, get_transforms
from models.unet import UNet
from metrics.dice_loss import DiceLoss, dice_score
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
    
    week_transforms = get_transforms(params, mode='week')
    strong_transforms = get_transforms(params, mode='strong')
    totensor = ToTensor()
    toimage = ToImage()
    train_transforms = Compose([week_transforms, totensor])
    params.week_transforms = week_transforms
    params.strong_transforms = strong_transforms
    
    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    labeled_dataset = BrainSegmentationDataset(data_dir, subset="labeled", transform=train_transforms, params=params)
    unlabeled_dataset = BrainSegmentationDataset(data_dir, subset="unlabeled", transform=train_transforms, params=params)
    val_dataset = BrainSegmentationDataset(data_dir, subset="validation", transform=totensor, params=params)
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=params.batch_ratio*params.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True, drop_last=False)
    
    model = UNet(in_channels=3, out_channels=1)
    model.to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = DiceLoss()
    metrics = {'dice score': dice_score}
    
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

    