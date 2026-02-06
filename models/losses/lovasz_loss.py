"""
Lovasz Loss for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import jittor as jt
from jittor import nn

from .utils import weight_reduce_loss


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.
    
    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float32().cumsum(0)
    union = gts + (1 - gt_sorted).float32().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_binary_logits(logits, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case).
    
    Remove labels equal to 'ignore_index'.
    """
    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return logits, labels
    valid = labels != ignore_index
    vlogits = logits[valid]
    vlabels = labels[valid]
    return vlogits, vlabels


def flatten_probs(probs, labels, ignore_index=None):
    """Flattens predictions in the batch."""
    if probs.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C=P,C
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = labels != ignore_index
    
    # Get valid indices
    valid_indices = jt.nonzero(valid).squeeze()
    if valid_indices.numel() == 0:
        return probs[:0], labels[:0]
    
    vprobs = probs[valid_indices]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.

    Args:
        logits (jt.Var): [P], logits at each prediction
            (between -infty and +infty).
        labels (jt.Var): [P], binary ground truth labels (0 or 1).

    Returns:
        jt.Var: The calculated loss.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    
    signs = 2.0 * labels.float32() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = jt.argsort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = (jt.relu(errors_sorted) * grad).sum()
    return loss


def lovasz_hinge(logits,
                labels,
                classes='present',
                per_image=False,
                class_weight=None,
                reduction='mean',
                avg_factor=None,
                ignore_index=255):
    """Binary Lovasz hinge loss.

    Args:
        logits (jt.Var): [B, H, W], logits at each pixel
            (between -infty and +infty).
        labels (jt.Var): [B, H, W], binary ground truth masks (0 or 1).
        classes (str | list[int], optional): Placeholder, to be consistent with
            other loss. Default: None.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): Placeholder, to be consistent
            with other loss. Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): The label index to be ignored. Default: 255.

    Returns:
        jt.Var: The calculated loss.
    """
    if per_image:
        loss = []
        for logit, label in zip(logits, labels):
            flat_logit, flat_label = flatten_binary_logits(
                logit.unsqueeze(0), label.unsqueeze(0), ignore_index)
            loss.append(lovasz_hinge_flat(flat_logit, flat_label))
        loss = weight_reduce_loss(jt.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_logits(logits, labels, ignore_index))
    return loss


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (jt.Var): [P, C], class probabilities at each prediction
            (between 0 and 1).
        labels (jt.Var): [P], ground truth labels (between 0 and C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.

    Returns:
        jt.Var: The calculated loss.
    """
    if probs.numel() == 0:
        # only void pixels, the gradients should be 0
        return probs * 0.0
    
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    
    for c in class_to_sum:
        fg = (labels == c).float32()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        
        errors = (fg - class_pred).abs()
        errors_sorted, perm = jt.argsort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        loss = (errors_sorted * lovasz_grad(fg_sorted)).sum()
        
        if class_weight is not None:
            loss *= class_weight[c]
        losses.append(loss)
    
    return jt.stack(losses).mean()


def lovasz_softmax(probs,
                  labels,
                  classes='present',
                  per_image=False,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=255):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (jt.Var): [B, C, H, W], class probabilities at each
            prediction (between 0 and 1).
        labels (jt.Var): [B, H, W], ground truth labels (between 0 and
            C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): The label index to be ignored. Default: 255.

    Returns:
        jt.Var: The calculated loss.
    """

    if per_image:
        loss = []
        for prob, label in zip(probs, labels):
            flat_prob, flat_label = flatten_probs(
                prob.unsqueeze(0), label.unsqueeze(0), ignore_index)
            loss.append(lovasz_softmax_flat(
                flat_prob, flat_label, classes=classes, class_weight=class_weight))
        loss = weight_reduce_loss(jt.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(
            *flatten_probs(probs, labels, ignore_index), 
            classes=classes, class_weight=class_weight)
    return loss


class LovaszLoss(nn.Module):
    """LovaszLoss implementation for Jittor.

    This loss is proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks <https://arxiv.org/abs/1705.08790>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: True.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". Default: 'mean'.
        class_weight (list[float], optional): Weight of each class. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. Defaults to 'loss_lovasz'.
    """

    def __init__(self,
                 loss_type='multi_class',
                 classes='present',
                 per_image=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_lovasz'):
        super(LovaszLoss, self).__init__()
        assert loss_type in ('binary', 'multi_class'), (
            "loss_type should be 'binary' or 'multi_class'.")

        if loss_type == 'binary':
            self.cls_criterion = lovasz_hinge
        else:
            self.cls_criterion = lovasz_softmax
        
        assert classes in ('all', 'present') or isinstance(classes, list)
        if not per_image and reduction != 'none':
            print(f"Warning: When per_image=False, reduction should be 'none'. Setting reduction to 'none'.")
            reduction = 'none'

        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self._loss_name = loss_name

    def execute(self, cls_score, label, weight=None, avg_factor=None, 
                reduction_override=None, **kwargs):
        """Forward function.
        
        Args:
            cls_score (jt.Var): The prediction.
            label (jt.Var): The learning label of the prediction.
            weight (jt.Var, optional): Sample-wise loss weight.
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

        # if multi-class loss, transform logits to probs
        if self.cls_criterion == lovasz_softmax:
            cls_score = nn.softmax(cls_score, dim=1)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.classes,
            self.per_image,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name 