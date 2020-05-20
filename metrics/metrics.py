
#pylint: disable=invalid-name

import numpy as np
import torch

SMOOTH = 1e-6

from sklearn.metrics import confusion_matrix  
from sklearn.metrics import jaccard_score  
import numpy as np

def compute_iou(y_pred, y_true):
    ''' Evaluates mean IoU between prediction and ground truth '''
    #ignore_mask = (y != ignore_index)
    #y_pred, y = y_pred[ignore_mask], y[ignore_mask]
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.round(y_true).astype(int)
    y_pred = np.ndarray.flatten(y_pred)
    y_true = np.ndarray.flatten(y_true)
    #import pdb; pdb.set_trace()
    miou = np.mean(jaccard_score(y_true, y_pred,average=None))
    #miou = 0
    #for i in range(y_pred.shape[0]):
     #   miou += jaccard_score(y_true[i,:,:], y_pred[i,:,:])
    return miou
    """
    y_pred = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred, num_classes)
    true_pos = np.diag(conf_matrix)
    false_pos = np.sum(conf_matrix, dim=0) - true_pos
    false_neg = np.sum(conf_matrix, dim=1) - true_pos
    tp_fp_fn = true_pos + false_pos + false_neg

    exist_class_mask = tp_fp_fn > 0
    true_pos, tp_fp_fn = true_pos[exist_class_mask], tp_fp_fn[exist_class_mask]
    return np.mean(true_pos / tp_fp_fn)
    """
def iou_numpy(outputs: np.array, labels: np.array):
    #outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()

def mean_iou(y_pred, y, num_classes=19, ignore_index=255):
    ''' Evaluates mean IoU between prediction and ground truth '''
    #ignore_mask = (y != ignore_index)
    #y_pred, y = y_pred[ignore_mask], y[ignore_mask]

    conf_matrix = _confusion_matrix(y, y_pred, num_classes)
    true_pos = torch.diag(conf_matrix)
    false_pos = torch.sum(conf_matrix, dim=0) - true_pos
    false_neg = torch.sum(conf_matrix, dim=1) - true_pos
    tp_fp_fn = true_pos + false_pos + false_neg

    exist_class_mask = tp_fp_fn > 0
    true_pos, tp_fp_fn = true_pos[exist_class_mask], tp_fp_fn[exist_class_mask]
    return torch.mean(true_pos / tp_fp_fn)

def pixel_accuracy(y_pred, y, num_classes, ignore_index=255):
    ''' Evaluates pixel accuracy between prediction and ground truth '''
    mask = (y != ignore_index)
    y_pred, y = y_pred[mask], y[mask]

    conf_matrix = _confusion_matrix(y, y_pred, num_classes)
    return torch.sum(torch.diag(conf_matrix)) / torch.sum(conf_matrix)

# Helper functions

def _one_hot(labels, num_classes, class_dim=1):
    ''' Converts a labels tensor (NHW) into a one-hot tensor (NLHW) '''
    labels = torch.unsqueeze(labels, class_dim)
    labels_one_hot = torch.zeros_like(labels).repeat(
        [num_classes if d == class_dim else 1
         for d in range(len(labels.shape))])
    labels_one_hot.scatter_(class_dim, labels, 1)
    return labels_one_hot

def _confusion_matrix(y_pred, y, num_classes):
    ''' Computes the confusion matrix between two predicitons '''
    b_size = y_pred.shape[0]
    y, y_pred = _one_hot(y, num_classes), _one_hot(y_pred, num_classes)
    y, y_pred = y.reshape(b_size, num_classes, -1), y_pred.reshape(b_size, num_classes, -1)
    return torch.einsum('iaj,ibj->ab', y.float(), y_pred.float())
