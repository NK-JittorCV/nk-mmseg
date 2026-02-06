"""
Dice Loss for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import jittor as jt
from jittor import nn

from .utils import weighted_loss, weight_reduce_loss


@weighted_loss
def dice_loss(pred, target, valid_mask, smooth=1, exponent=2, class_weight=None, ignore_index=255):
    """Calculate dice loss.
    
    Args:
        pred (jt.Var): Prediction tensor with shape (N, C, H, W).
        target (jt.Var): Target tensor with shape (N, C, H, W).
        valid_mask (jt.Var): Valid mask tensor.
        smooth (float): Smoothing parameter to avoid division by zero.
        exponent (float): Exponent for dice calculation.
        class_weight (jt.Var, optional): Class weights.
        ignore_index (int): Index to ignore.
    """
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss_i = binary_dice_loss(
                pred[:, i], target[:, i], valid_mask=valid_mask, 
                smooth=smooth, exponent=exponent
            )
            if class_weight is not None:
                dice_loss_i *= class_weight[i]
            total_loss += dice_loss_i
    
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    """Calculate binary dice loss.
    
    Args:
        pred (jt.Var): Prediction tensor.
        target (jt.Var): Target tensor.
        valid_mask (jt.Var): Valid mask tensor.
        smooth (float): Smoothing parameter.
        exponent (float): Exponent for dice calculation.
    """
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = jt.sum(jt.multiply(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = jt.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


class DiceLoss(nn.Module):
    """Dice Loss implementation for Jittor.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". Default: 'mean'.
        class_weight (list[float], optional): Weight of each class. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def execute(self, pred, target, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function.
        
        Args:
            pred (jt.Var): The prediction.
            target (jt.Var): The learning label of the prediction.
            avg_factor (int, optional): Average factor that is used to average the loss.
            reduction_override (str, optional): The reduction method used to override.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
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

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index
        )
        return loss

    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name 