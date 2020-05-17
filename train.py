import os
import logging
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from datasets.transforms import ToTensor, ToImage
from evaluate import evaluate
from utils.params import Params, save_dict_to_json, load_dict_to_json
from utils.logger import RunningAverage, set_logger, save_checkpoint, load_checkpoint
from utils.visualizer import normalize_tensor, plot_samples


def supervised(model, loss_fn, lab_image_batch, lab_mask_batch, unl_image_batch, params):
    """Supervised training algorithm
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        lab_image_batch: (torch.tensor) a batch of labeled images
        lab_mask_batch: (torch.tensor) a batch of masks
        unl_image_batch: (torch.tensor) a batch of unlabeled images
        params: (Params) hyperparameters
    """
    
    lab_pred_batch = model(lab_image_batch)
    lab_loss = loss_fn(lab_pred_batch, lab_mask_batch)
    
    unl_loss = torch.zeros(1).to(params.device)
    return lab_loss, unl_loss


def semi_supervised(model, loss_fn, lab_image_batch, lab_mask_batch, unl_image_batch, params):
    """Semi-supervised training algorithm
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        lab_image_batch: (torch.tensor) a batch of labeled images
        lab_mask_batch: (torch.tensor) a batch of masks
        unl_image_batch: (torch.tensor) a batch of unlabeled images
        params: (Params) hyperparameters
    """
    
    lab_pred_batch = model(lab_image_batch)
    lab_loss = loss_fn(lab_pred_batch, lab_mask_batch)
    
    if lab_loss.item() > 0.5:
        unl_loss = torch.zeros(1).to(params.device)
    else:
        unl_pred_batch = model(unl_image_batch)
        new_image_batch, new_mask_batch = torch.zeros_like(unl_image_batch), torch.zeros_like(unl_pred_batch)
        c = 0
        for b in range(params.batch_ratio*params.batch_size):
            b_pred = unl_pred_batch[b]
            if ((b_pred > 0.5).any() and b_pred[b_pred > 0.5].mean() > 0.99) or (not (b_pred > 0.5).any()):
                if (b_pred < 0.5).any() and b_pred[b_pred < 0.5].mean() < 0.01:
                    b_mask = (b_pred > 0.5).float()
                    b_image, b_mask = ToImage()((unl_image_batch[b], b_mask))
                    b_image, b_mask = params.strong_transforms((b_image, b_mask))
                    b_image, b_mask = ToTensor()((b_image, b_mask))
                    new_image_batch[c], new_mask_batch[c] = b_image, b_mask
                    c += 1
        new_image_batch, new_mask_batch = new_image_batch.to(params.device), new_mask_batch.to(params.device)
        new_pred_batch = model(new_image_batch)
        unl_loss = loss_fn(new_pred_batch[:c, ...], new_mask_batch[:c, ...])

        if params.step % params.save_summary_steps == 0:
            new_image_grid = normalize_tensor(torchvision.utils.make_grid(new_image_batch, nrow=4))
            new_pred_grid = torchvision.utils.make_grid(new_pred_batch, nrow=4)
            new_mask_grid = torchvision.utils.make_grid(new_mask_batch, nrow=4)
            samples_grid = {'consistency/image': new_image_grid, 'consistency/pred': new_pred_grid, 
                    'consistency/mask': new_mask_grid}
            for sample in samples_grid:
                params.writer.add_image(sample, samples_grid[sample], params.step)
    return lab_loss, unl_loss


def consistency(model, loss_fn, lab_image_batch, lab_mask_batch, unl_image_batch, params):
    """Semi-supervised training algorithm with consistency loss
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        lab_image_batch: (torch.tensor) a batch of labeled images
        lab_mask_batch: (torch.tensor) a batch of masks
        unl_image_batch: (torch.tensor) a batch of unlabeled images
        params: (Params) hyperparameters
    """
    
    lab_pred_batch = model(lab_image_batch)
    lab_loss = loss_fn(lab_pred_batch, lab_mask_batch)
    
    if lab_loss.item() > 0.5:
        unl_loss = torch.zeros(1).to(params.device)
    else:
        unl_pred_batch = model(unl_image_batch)
        new_image_batch, new_mask_batch = torch.zeros_like(unl_image_batch), torch.zeros_like(unl_pred_batch)
        for b in range(params.batch_ratio*params.batch_size):
            b_image, b_mask = ToImage()((unl_image_batch[b], unl_pred_batch[b]))
            b_image, b_mask = params.strong_transforms((b_image, b_mask))
            b_image, b_mask = ToTensor()((b_image, b_mask))
            new_image_batch[b], new_mask_batch[b] = b_image, b_mask
        new_image_batch, new_mask_batch = new_image_batch.to(params.device), new_mask_batch.to(params.device)
        new_pred_batch = model(new_image_batch)
        unl_loss = loss_fn(new_pred_batch, new_mask_batch)

        if params.step % params.save_summary_steps == 0:
            new_image_grid = normalize_tensor(torchvision.utils.make_grid(new_image_batch, nrow=4))
            new_pred_grid = torchvision.utils.make_grid(new_pred_batch, nrow=4)
            new_mask_grid = torchvision.utils.make_grid(new_mask_batch, nrow=4)
            samples_grid = {'consistency/image': new_image_grid, 'consistency/pred': new_pred_grid, 
                    'consistency/mask': new_mask_grid}
            for sample in samples_grid:
                params.writer.add_image(sample, samples_grid[sample], params.step)
    return lab_loss, unl_loss


def confidence_map(model, loss_fn, lab_image_batch, lab_mask_batch, unl_image_batch, params):
    """Semi-supervised training algorithm
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        lab_image_batch: (torch.tensor) a batch of labeled images
        lab_mask_batch: (torch.tensor) a batch of masks
        unl_image_batch: (torch.tensor) a batch of unlabeled images
        params: (Params) hyperparameters
    """
    
    lab_pred_batch = model(lab_image_batch)
    lab_loss = loss_fn(lab_pred_batch, lab_mask_batch)
    
    if lab_loss.item() > 0.5:
        unl_loss = torch.zeros(1).to(params.device)
    else:
        unl_pred_batch = model(unl_image_batch)
        new_image_batch, new_mask_batch = torch.zeros_like(unl_image_batch), torch.zeros_like(unl_pred_batch)
        conf_map_batch = torch.zeros_like(unl_pred_batch)
        for b in range(params.batch_ratio*params.batch_size):
            b_image, b_heatmap = ToImage()((unl_image_batch[b], unl_pred_batch[b]))
            b_image, b_heatmap = params.strong_transforms((b_image, b_heatmap))
            b_conf = ((b_heatmap > 0.99) + (b_heatmap < 0.01)).astype(np.float32)
            b_mask = (b_heatmap > 0.5).astype(np.float32)
            _, b_conf = ToTensor()((b_image, b_conf))
            b_image, b_mask = ToTensor()((b_image, b_mask))
            new_image_batch[b], new_mask_batch[b] = b_image, b_mask
            conf_map_batch[b] = b_conf
        new_image_batch, new_mask_batch = new_image_batch.to(params.device), new_mask_batch.to(params.device)
        new_pred_batch = model(new_image_batch)
        unl_loss = loss_fn(conf_map_batch * new_pred_batch, conf_map_batch * new_mask_batch)

        if params.step % params.save_summary_steps == 0:
            new_image_grid = normalize_tensor(torchvision.utils.make_grid(new_image_batch, nrow=4))
            new_pred_grid = torchvision.utils.make_grid(new_pred_batch, nrow=4)
            new_mask_grid = torchvision.utils.make_grid(new_mask_batch, nrow=4)
            new_conf_grid = torchvision.utils.make_grid(conf_map_batch, nrow=4)
            samples_grid = {'consistency/image': new_image_grid, 'consistency/pred': new_pred_grid, 
                    'consistency/mask': new_mask_grid, 'consistency/conf': new_conf_grid}
            for sample in samples_grid:
                params.writer.add_image(sample, samples_grid[sample], params.step)
    return lab_loss, unl_loss


def train_epoch(algorithm, model, optimizer, loss_fn, labeled_loader, unlabeled_loader, metrics, params):
    """Train the model for one epoch
    Args:
        algorithm: (string) algorithm to use for training e.g. 'supervised', 'semi-supervised'
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        labeled_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches labeled data
        unlabeled_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches unlabeled data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """
    
    model.train()
    
    labeled_iter = iter(labeled_loader)
    
    summ = []
    loss_avg = RunningAverage()
    
    with tqdm(total=len(unlabeled_loader)) as pbar:
        for i, (unl_image_batch, unl_mask_batch) in enumerate(unlabeled_loader):
            params.step = (params.epoch - 1) * len(unlabeled_loader) + i
            
            try:
                lab_image_batch, lab_mask_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                lab_image_batch, lab_mask_batch = next(labeled_iter)
            
            lab_image_batch, lab_mask_batch, unl_image_batch, unl_mask_batch, = lab_image_batch.to(
                params.device, non_blocking=True), lab_mask_batch.to(
                params.device, non_blocking=True), unl_image_batch.to(
                params.device, non_blocking=True), unl_mask_batch.to(
                params.device, non_blocking=True)
            
            lab_pred_batch = model(lab_image_batch)
            unl_pred_batch = model(unl_image_batch)
            
            if algorithm == 'supervised':
                lab_loss, unl_loss = supervised(model, loss_fn, lab_image_batch, lab_mask_batch, unl_image_batch, params)
            elif algorithm == 'semi-supervised':
                lab_loss, unl_loss = semi_supervised(model, loss_fn, lab_image_batch, lab_mask_batch, unl_image_batch, params)
            elif algorithm == 'confidence-map':
                lab_loss, unl_loss = confidence_map(model, loss_fn, lab_image_batch, lab_mask_batch, unl_image_batch, params)
            elif algorithm == 'consistency':
                lab_loss, unl_loss = consistency(model, loss_fn, lab_image_batch, lab_mask_batch, unl_image_batch, params)
            else:
                raise NotImplemented
            
            loss = lab_loss + unl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if params.step % params.save_summary_steps == 0:
                lab_pred_batch_np = lab_pred_batch.detach().cpu().numpy()
                lab_mask_batch_np = lab_mask_batch.detach().cpu().numpy()
                summary_batch = {metric: metrics[metric](lab_pred_batch_np, lab_mask_batch_np) for metric in metrics}
                summary_batch['labeled loss'] = lab_loss.item()
                summary_batch['unlabeled loss'] = unl_loss.item()
                summary_batch['loss'] = loss.item()
                for metric in summary_batch:
                    tag = metric.replace(" ", "_")
                    tag = ('loss/' + tag) if 'loss' in tag else tag
                    params.writer.add_scalars(tag, {'train': summary_batch[metric]}, params.step)
                summ.append(summary_batch)
            
            loss_avg.update(loss.item())
            pbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            pbar.update()
    
    if len(summ):
        metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)
    else:
        metrics_mean = None
    
    lab_image_grid = normalize_tensor(torchvision.utils.make_grid(lab_image_batch, nrow=4))
    unl_image_grid = normalize_tensor(torchvision.utils.make_grid(unl_image_batch, nrow=4))
    lab_pred_grid = torchvision.utils.make_grid(lab_pred_batch, nrow=4)
    unl_pred_grid = torchvision.utils.make_grid(unl_pred_batch, nrow=4)
    lab_mask_grid = torchvision.utils.make_grid(lab_mask_batch, nrow=4)
    unl_mask_grid = torchvision.utils.make_grid(unl_mask_batch, nrow=4)
    samples_grid = {'images/train_lab': lab_image_grid, 'preds/train_lab': lab_pred_grid, 
                    'masks/train_lab': lab_mask_grid, 'images/train_unl': unl_image_grid, 
                    'preds/train_unl': unl_pred_grid, 'masks/train_unl': unl_mask_grid}
    for sample in samples_grid:
        params.writer.add_image(sample, samples_grid[sample], params.step)
    return metrics_mean, samples_grid
            

def train_eval(algorithm, model, optimizer, loss_fn, labeled_loader, unlabeled_loader, val_loader, metrics, params):
    """Train the model and evaluate it at tha same time
    Args:
        algorithm: (string) algorithm to use for training e.g. 'supervised', 'semi-supervised'
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        labeled_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches labeled data
        unlabeled_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches unlabeled data
        val_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """
    
    epoch = 0
    best_val_score = 0
    
    if params.restore:
        restore_path = os.path.join(params.model_dir, 'checkpoints', 'best.pth.tar')
        if os.path.exists(restore_path):
            logging.info("Restoring parameters from {}".format(restore_path))
            checkpoint = load_checkpoint(restore_path, model, optimizer)
            epoch = checkpoint['epoch']
        metrics_path = os.path.join(params.model_dir, "metrics.json")
        if os.path.exists(metrics_path):
            val_metrics = {}
            val_metrics = load_dict_to_json(val_metrics, metrics_path)
            best_val_score = val_metrics['dice score']
            
    log_path = os.path.join(params.model_dir, 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = SummaryWriter(log_path)
    params.writer = writer
    
    num_epochs = params.num_epochs
    num_epochs = num_epochs * int(1 + len(labeled_loader)/len(unlabeled_loader))
    while epoch < num_epochs:
        epoch += 1
        params.epoch = epoch
        logging.info("Epoch {}/{}".format(epoch, num_epochs))
        
        train_metrics, train_samples = train_epoch(algorithm, model, optimizer, loss_fn, labeled_loader, 
                                                   unlabeled_loader,metrics, params)
        
        val_metrics, val_samples = evaluate(model, loss_fn, val_loader, metrics, params)
        
        for metric in val_metrics:
            tag = metric.replace(" ", "_")
            tag = ('loss/' + tag) if 'loss' in tag else tag
            writer.add_scalars(tag,
                               {'validation': val_metrics[metric]},
                                epoch * len(unlabeled_loader))
        for sample in val_samples:
             writer.add_image(sample, val_samples[sample], epoch * len(unlabeled_loader))
        
        val_score = val_metrics['dice score']
        is_best = val_score >= best_val_score
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint=os.path.join(params.model_dir, 'checkpoints'))
        if is_best:
            best_val_score = val_score
            save_dict_to_json(val_metrics, os.path.join(params.model_dir, "metrics.json"))
            
        writer.close()    
