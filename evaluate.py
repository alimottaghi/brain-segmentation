import os
import logging
import numpy as np
import torch
from utils.visualizer import save_image_mask


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    model.eval()

    summ = []

    for image_batch, mask_batch in dataloader:

        image_batch, mask_batch = image_batch.to(
                params.device, non_blocking=True), mask_batch.to(params.device, non_blocking=True)
        
        with torch.set_grad_enabled(False):
            pred_batch = model(image_batch)
            loss = loss_fn(pred_batch, mask_batch)

            pred_batch = pred_batch.detach().cpu().numpy()
            mask_batch = mask_batch.detach().cpu().numpy()
            summary_batch = {metric: metrics[metric](pred_batch, mask_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def generate_outputs(model, dataloader, params, save=True):
    """Generate output predictions of the model
    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
        save: (Boolean) whther to save the images or not
    """
    
    model.eval()

    image_list = []
    mask_list = []
    pred_list = []
    for i, (image_batch, mask_batch) in enumerate(dataloader):
        image_batch_device, mask_batch_device = image_batch.to(params.device), mask_batch.to(params.device)
        batch_size = len(image_batch)
        with torch.set_grad_enabled(False):
            pred_batch_device = model(image_batch_device)
            pred_batch = pred_batch_device.detach().cpu()
            pred_list.extend([np.round(pred_batch[s].numpy().transpose(1, 2, 0)).astype(int) for s in range(batch_size)])
            image_list.extend([image_batch[s].numpy().transpose(1, 2, 0) for s in range(batch_size)])
            mask_list.extend([mask_batch[s].numpy().transpose(1, 2, 0) for s in range(batch_size)])

    output_list = []
    os.makedirs(os.path.join(params.model_dir, "output"), exist_ok=True)
    for i in range(len(image_list)):
        sample = (image_list[i], mask_list[i], pred_list[i])
        output_list.append(sample)
        if save:
            image_path = os.path.join(params.model_dir, "output/val_{}.png".format(i))
            save_image_mask(image_path, *sample)
    return output_list