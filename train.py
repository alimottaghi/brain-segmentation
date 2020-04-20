import os
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from evaluate import evaluate
from utils.params import Params, save_dict_to_json, load_dict_to_json
from utils.logger import RunningAverage, set_logger, save_checkpoint, load_checkpoint


def train_supervised(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """
    
    model.train()
    
    summ = []
    loss_avg = RunningAverage()
    
    with tqdm(total=len(dataloader)) as pbar:
        for i, (image_batch, mask_batch) in enumerate(dataloader):
            
            image_batch, mask_batch = image_batch.to(
                params.device, non_blocking=True), mask_batch.to(params.device, non_blocking=True)
            
            pred_batch = model(image_batch)
            loss = loss_fn(pred_batch, mask_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % params.save_summary_steps == 0:
                pred_batch = pred_batch.detach().cpu().numpy()
                mask_batch = mask_batch.detach().cpu().numpy()
                summary_batch = {metric: metrics[metric](pred_batch, mask_batch) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)
            
            loss_avg.update(loss.item())
            pbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            pbar.update()
    
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def train_eval_supervised(model, optimizer, loss_fn, train_loader, val_loader, metrics, params):
    """Train the model and evaluate it at tha same time
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        train_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
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

    while epoch < params.num_epochs:
        epoch += 1
        logging.info("Epoch {}/{}".format(epoch, params.num_epochs))
        
        train_metrics = train_supervised(model, optimizer, loss_fn, train_loader, metrics, params)
        
        val_metrics = evaluate(model, loss_fn, val_loader, metrics, params)
        
        for metric in val_metrics:
            writer.add_scalars(metric.replace(" ", "_"),
                              {'train': train_metrics[metric], 
                              'validation': val_metrics[metric]},
                              epoch * len(train_loader))
        
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


def train_semi_supervised(model, optimizer, loss_fn, labeled_loader, unlabeled_loader, metrics, params):
    """Train the model
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        labeled_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches labeled data
        unlabeled_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches unlabeled data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """
    
    model.train()
    
    unlabeled_iter = iter(unlabeled_loader)
    
    summ = []
    loss_avg = RunningAverage()
    lab_loss_avg = RunningAverage()
    unl_loss_avg = RunningAverage()
    
    with tqdm(total=len(labeled_loader)) as pbar:
        for i, (lab_image_batch, lab_mask_batch) in enumerate(labeled_loader):
            
            try:
                unl_image_batch, unl_mask_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unl_image_batch, unl_mask_batch = next(unlabeled_iter)
            
            lab_image_batch, lab_mask_batch, unl_image_batch = lab_image_batch.to(
                params.device, non_blocking=True), lab_mask_batch.to(
                params.device, non_blocking=True), unl_image_batch.to(params.device, non_blocking=True)
            
            lab_pred_batch = model(lab_image_batch)
            unl_pred_batch = model(unl_image_batch)
            
            lab_loss = loss_fn(lab_pred_batch, lab_mask_batch)
            
            unl_loss = torch.zeros(1).to(params.device)
            unl_counter = torch.zeros(1).to(params.device)
            for b in range(params.mu*params.batch_size):
                b_pred = unl_pred_batch[b]
                if (b_pred > 0.5).any() and b_pred[b_pred > 0.5].mean() > 0.99:
                    if (b_pred < 0.5).any() and b_pred[b_pred < 0.5].mean() < 0.01:
                        b_mask = (b_pred > 0.5).detach().cpu().numpy().transpose(1, 2, 0)
                        b_image = unl_image_batch[b].detach().cpu().numpy().transpose(1, 2, 0)
                        b_image, b_mask = params.strong_transforms((b_image, b_mask))
                        b_image, b_mask = b_image.to(params.device).unsqueeze(0), b_mask.to(params.device).unsqueeze(0)
                        b_pred = model(b_image)
                        unl_loss += loss_fn(b_pred, b_mask)
                        unl_counter += 1
            
            if unl_counter.item() > 0:
                unl_loss = unl_loss / unl_counter
            
            loss = lab_loss + unl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % params.save_summary_steps == 0:
                lab_pred_batch = lab_pred_batch.detach().cpu().numpy()
                lab_mask_batch = lab_mask_batch.detach().cpu().numpy()
                summary_batch = {metric: metrics[metric](lab_pred_batch, lab_mask_batch) for metric in metrics}
                summary_batch['labeled loss'] = lab_loss.item()
                summary_batch['unlabeled loss'] = unl_loss.item()
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)
            
            loss_avg.update(loss.item())
            lab_loss_avg.update(lab_loss.item())
            unl_loss_avg.update(unl_loss.item())
            pbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            pbar.update()
    
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean
            

def train_eval_semi_supervised(model, optimizer, loss_fn, labeled_loader, unlabeled_loader, val_loader, metrics, params):
    """Train the model and evaluate it at tha same time
    Args:
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
    
    while epoch < params.num_epochs:
        epoch += 1
        logging.info("Epoch {}/{}".format(epoch, params.num_epochs))
        
        train_metrics = train_semi_supervised(model, optimizer, loss_fn, labeled_loader, unlabeled_loader, metrics, params)
        
        val_metrics = evaluate(model, loss_fn, val_loader, metrics, params)
        
        for metric in val_metrics:
            writer.add_scalars(metric.replace(" ", "_"),
                              {'train': train_metrics[metric], 
                              'validation': val_metrics[metric]},
                              epoch * len(labeled_loader))
        
        writer.add_scalars('loss/unlabeled_loss',
                              {'train': train_metrics['unlabeled loss']},
                              epoch * len(labeled_loader))
        
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