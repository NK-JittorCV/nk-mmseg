"""
Cross entropy loss for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import warnings
import jittor as jt
from jittor import nn

from .utils import get_class_weight, weight_reduce_loss


def cross_entropy(
    pred,
    label,
    weight=None,
    class_weight=None,
    reduction="mean",
    avg_factor=None,
    ignore_index=255,
    avg_non_ignore=False,
):
    """Cross entropy loss function.
    
    Args:
        pred (jt.Var): The prediction with shape (N, C, H, W).
        label (jt.Var): The learning label of the prediction.
        weight (jt.Var, optional): Sample-wise loss weight.
        class_weight (list[float], optional): The weight for each class.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average the loss.
        ignore_index (int): Specifies a target value that is ignored.
        avg_non_ignore (bool): Whether the loss is only averaged over non-ignored targets.
    """
    # Apply cross entropy loss
    loss = nn.cross_entropy_loss(pred, label, ignore_index=ignore_index)
    
    # Apply class weights if provided
    if class_weight is not None:
        if isinstance(class_weight, list):
            class_weight = jt.array(class_weight).float32()
        
        # Expand class weights to match prediction shape
        if len(pred.shape) == 4:  # (N, C, H, W)
            N, C, H, W = pred.shape
            class_weight = class_weight.view(1, C, 1, 1).expand(N, C, H, W)
        
        # Apply weights
        pred_softmax = nn.softmax(pred, dim=1)
        weighted_pred = pred_softmax * class_weight
        loss = nn.cross_entropy_loss(weighted_pred, label, ignore_index=ignore_index)
    
    # Apply sample weights and reduction
    if weight is not None:
        loss = loss * weight
    
    if reduction == "mean":
        if avg_non_ignore and ignore_index is not None:
            # Average over non-ignored elements
            valid_mask = (label != ignore_index)
            if valid_mask.sum() > 0:
                loss = loss.sum() / valid_mask.sum()
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass  # Keep original shape
    
    return loss


def binary_cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=255,
    avg_non_ignore=False,
):
    """Calculate the binary CrossEntropy loss.
    
    Args:
        pred (jt.Var): The prediction with shape (N, 1, H, W).
        label (jt.Var): The learning label of the prediction.
        weight (jt.Var, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average the loss.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): Specifies a target value that is ignored.
        avg_non_ignore (bool): Whether the loss is only averaged over non-ignored targets.
    """
    # Convert to binary classification format
    if pred.shape[1] > 1:
        # Multi-class to binary
        pred = nn.sigmoid(pred)
    
    # Create binary labels
    binary_label = (label != ignore_index).float32()
    
    # Apply binary cross entropy
    loss = nn.binary_cross_entropy_with_logits(pred.squeeze(1), binary_label)
    
    # Apply weights and reduction
    if weight is not None:
        loss = loss * weight
    
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    
    return loss


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss module.
    
    Args:
        use_sigmoid (bool, optional): Whether to use sigmoid activation.
        use_mask (bool, optional): Whether to use mask.
        reduction (str, optional): The method used to reduce the loss.
        class_weight (list[float], optional): Weight of each class.
        ignore_index (int, optional): The label index to be ignored.
        loss_weight (float, optional): Weight of this loss item.
        avg_non_ignore (bool): Whether to average loss over non-ignored targets.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=255,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def execute(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, dtype=jt.float32)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls


def mask_cross_entropy(pred,
                      target,
                      label,
                      reduction='mean',
                      avg_factor=None,
                      class_weight=None,
                      ignore_index=None,
                      **kwargs):
    """Calculate the CrossEntropy loss for masks.
    
    Args:
        pred (jt.Var): The prediction with shape (N, C, H, W).
        target (jt.Var): The learning label of the prediction.
        label (jt.Var): ``label`` indicates the class label of the sample.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average the loss.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int, optional): The label index to be ignored.
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = jt.arange(0, num_rois, dtype=jt.int64)
    pred_slice = pred[inds, label].squeeze(1)
    return nn.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction=reduction)[None]
