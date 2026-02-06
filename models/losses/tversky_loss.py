"""
Tversky Loss for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import jittor as jt
from jittor import nn

from .utils import weighted_loss


@weighted_loss
def tversky_loss(pred, target, valid_mask, alpha=0.3, beta=0.7, smooth=1, 
                class_weight=None, ignore_index=255):
    """Calculate tversky loss.
    
    Args:
        pred (jt.Var): Prediction tensor with shape (N, C, H, W).
        target (jt.Var): Target tensor with shape (N, C, H, W).
        valid_mask (jt.Var): Valid mask tensor.
        alpha (float): The coefficient of false positives.
        beta (float): The coefficient of false negatives.
        smooth (float): Smoothing parameter to avoid division by zero.
        class_weight (jt.Var, optional): Class weights.
        ignore_index (int): Index to ignore.
    """
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    
    for i in range(num_classes):
        if i != ignore_index:
            tversky_loss_i = binary_tversky_loss(
                pred[:, i], target[:, i], valid_mask=valid_mask,
                alpha=alpha, beta=beta, smooth=smooth
            )
            if class_weight is not None:
                tversky_loss_i *= class_weight[i]
            total_loss += tversky_loss_i
    
    return total_loss / num_classes


@weighted_loss
def binary_tversky_loss(pred, target, valid_mask, alpha=0.3, beta=0.7, smooth=1):
    """Calculate binary tversky loss.
    
    Args:
        pred (jt.Var): Prediction tensor.
        target (jt.Var): Target tensor.
        valid_mask (jt.Var): Valid mask tensor.
        alpha (float): The coefficient of false positives.
        beta (float): The coefficient of false negatives.
        smooth (float): Smoothing parameter.
    """
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = jt.sum(jt.multiply(pred, target) * valid_mask, dim=1)
    FP = jt.sum(jt.multiply(pred, 1 - target) * valid_mask, dim=1)
    FN = jt.sum(jt.multiply(1 - pred, target) * valid_mask, dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - tversky


class TverskyLoss(nn.Module):
    """Tversky Loss implementation for Jittor.
    
    This loss is proposed in `Tversky loss function for image
    segmentation using 3D fully convolutional deep networks.
    <https://arxiv.org/abs/1706.05721>`_.
    
    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1.
        class_weight (list[float], optional): Weight of each class. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int): The label index to be ignored. Default: 255.
        alpha(float, in [0, 1]): The coefficient of false positives. Default: 0.3.
        beta (float, in [0, 1]): The coefficient of false negatives. Default: 0.7.
            Note: alpha + beta = 1.
        loss_name (str, optional): Name of the loss item. Defaults to 'loss_tversky'.
    """

    def __init__(self,
                 smooth=1,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 alpha=0.3,
                 beta=0.7,
                 loss_name='loss_tversky'):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        assert alpha + beta == 1.0, 'Sum of alpha and beta must be 1.0!'
        self.alpha = alpha
        self.beta = beta
        self._loss_name = loss_name

    def execute(self, pred, target, **kwargs):
        """Forward function.
        
        Args:
            pred (jt.Var): The prediction.
            target (jt.Var): The learning label of the prediction.
        """
        if self.class_weight is not None:
            class_weight = jt.array(self.class_weight)
        else:
            class_weight = None

        pred = nn.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        
        # Create one-hot target
        target_clamped = jt.clamp(target.int64(), 0, num_classes - 1)
        one_hot_target = jt.zeros((target.shape[0], num_classes, *target.shape[1:]))
        
        # Fill one-hot target
        for i in range(num_classes):
            one_hot_target[:, i] = (target_clamped == i).float32()
        
        valid_mask = (target != self.ignore_index).int64()

        loss = self.loss_weight * tversky_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            alpha=self.alpha,
            beta=self.beta,
            smooth=self.smooth,
            class_weight=class_weight,
            ignore_index=self.ignore_index
        )
        return loss

    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name 