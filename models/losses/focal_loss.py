"""
Focal loss for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import jittor as jt
from jittor import nn

from .utils import weight_reduce_loss


def sigmoid_focal_loss(pred,
                      target,
                      weight=None,
                      gamma=2.0,
                      alpha=0.25,
                      reduction='mean',
                      avg_factor=None):
    """Sigmoid focal loss.
    
    Args:
        pred (jt.Var): The prediction with shape (N, C, H, W).
        target (jt.Var): The learning label of the prediction.
        weight (jt.Var, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating factor.
        alpha (float, optional): A balanced form for Focal Loss.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average the loss.
    """
    pred_sigmoid = jt.sigmoid(pred)
    target = target.float32()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = nn.binary_cross_entropy_with_logits(pred, target) * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                # which is broadcasted to (num_priors, num_classes).
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight could be of shape (num_priors, num_classes),
                # and it should be kept as is.
                assert weight.numel() == loss.numel()
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def softmax_focal_loss(pred,
                      target,
                      weight=None,
                      gamma=2.0,
                      alpha=0.25,
                      reduction='mean',
                      avg_factor=None,
                      ignore_index=255):
    """Softmax focal loss.
    
    Args:
        pred (jt.Var): The prediction with shape (N, C, H, W).
        target (jt.Var): The learning label of the prediction.
        weight (jt.Var, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating factor.
        alpha (float, optional): A balanced form for Focal Loss.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average the loss.
        ignore_index (int, optional): The label index to be ignored.
    """
    assert pred.dim() == 4, f"Expected 4D tensor, got {pred.dim()}D"
    assert target.dim() == 3, f"Expected 3D tensor, got {target.dim()}D"
    
    num_classes = pred.size(1)
    
    # Apply softmax to get probabilities
    pred_softmax = nn.softmax(pred, dim=1)
    
    # Build one-hot targets (N, C, H, W)
    target_clamped = jt.clamp(target.int64(), 0, num_classes - 1)
    target_one_hot = (target_clamped.unsqueeze(1) == jt.arange(num_classes).reshape(1, num_classes, 1, 1)).float32()
    
    # Mask out ignore_index
    if ignore_index is not None:
        valid_mask = (target != ignore_index).float32().unsqueeze(1)
        target_one_hot = target_one_hot * valid_mask
    
    # Calculate pt
    pt = (pred_softmax * target_one_hot).sum(dim=1)  # (N, H, W)
    
    # Avoid log(0)
    pt = jt.clamp(pt, 1e-8, 1.0)
    
    # Calculate cross entropy loss
    ce_loss = -jt.log(pt)
    
    # Calculate focal weight
    focal_weight = (1 - pt) ** gamma
    
    # Apply alpha weighting
    if alpha is not None:
        if isinstance(alpha, (list, tuple)):
            alpha_t = jt.array(alpha)[target]
        else:
            alpha_t = alpha
        focal_weight = alpha_t * focal_weight
    
    # Final focal loss
    loss = focal_weight * ce_loss
    
    # Apply sample weights and reduction
    if weight is not None:
        loss = loss * weight
    
    if reduction == 'mean':
        if ignore_index is not None:
            loss = loss.sum() / valid_mask.sum().float32()
        else:
            loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss


class FocalLoss(nn.Module):
    """Focal Loss module.
    
    Args:
        use_sigmoid (bool, optional): Whether to use sigmoid activation.
        gamma (float, optional): The gamma for calculating the modulating factor.
        alpha (float, optional): A balanced form for Focal Loss.
        reduction (str, optional): The method used to reduce the loss.
        loss_weight (float, optional): Weight of this loss item.
        ignore_index (int, optional): The label index to be ignored.
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255):
        super(FocalLoss, self).__init__()
        # Allow both sigmoid and softmax variants
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        
        Args:
            pred (jt.Var): The prediction.
            target (jt.Var): The learning label of the prediction.
            weight (jt.Var, optional): The weight of loss for each prediction.
            avg_factor (int, optional): Average factor that is used to average the loss.
            reduction_override (str, optional): The reduction method used to override.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            loss_cls = self.loss_weight * softmax_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
                ignore_index=self.ignore_index)
        return loss_cls


def py_sigmoid_focal_loss(pred,
                         target,
                         weight=None,
                         gamma=2.0,
                         alpha=0.25,
                         reduction='mean',
                         avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    
    Args:
        pred (jt.Var): The prediction with shape (N, C, H, W).
        target (jt.Var): The learning label of the prediction.
        weight (jt.Var, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating factor.
        alpha (float, optional): A balanced form for Focal Loss.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average the loss.
    """
    pred_sigmoid = nn.sigmoid(pred)
    target = target.float32()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = nn.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
