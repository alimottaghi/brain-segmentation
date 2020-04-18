import numpy as np
import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
    

def dice_score(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    sum_all = np.sum(y_pred) + np.sum(y_true)
    if sum_all > 0:
        return np.sum(y_pred[y_true == 1]) * 2.0 / sum_all
    else:
        return 1

