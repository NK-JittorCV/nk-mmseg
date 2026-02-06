"""
Various loss function implementations for DFormer Jittor.
"""

import numpy as np
import jittor as jt
from jittor import nn


class FocalLoss2d(nn.Module):
    """2D Focal Loss implementation."""
    
    def __init__(self, gamma=2.0, weight=None, reduction="mean", ignore_index=255):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            if isinstance(weight, list):
                weight = jt.array(weight).float32()
            self.loss = nn.NLLLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
        else:
            self.loss = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def execute(self, input, target):
        """Forward function."""
        softmax_pred = nn.softmax(input, dim=1)
        log_softmax_pred = nn.log_softmax(input, dim=1)
        weighted_pred = (1 - softmax_pred) ** self.gamma * log_softmax_pred
        return self.loss(weighted_pred, target)


class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss."""
    
    def __init__(self, ignore_index=255, reduction="mean", weight=None, class_num=37, beta=0.01):
        super().__init__()
        self.beta = beta
        self.class_num = class_num
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)
        self.criterion2 = nn.NLLLoss(reduction="none", ignore_index=ignore_index, weight=weight)

    def execute(self, pred, target):
        """Forward function."""
        b, c, h, w = pred.shape
        _, max_id = jt.argmax(pred, dim=1)
        target_flat = target.view(b, 1, h, w)
        mask = (target_flat != self.ignore_label).float32()
        target_flat = (mask * target_flat.float32()).int64()
        
        # Convert to onehot
        label_pred = jt.zeros((b, self.class_num, h, w)).scatter_(1, target_flat, 1.0)
        
        # Standard cross entropy loss
        weighted_pred = nn.log_softmax(pred, dim=1)
        loss1 = self.criterion(weighted_pred, target)
        
        # Reverse cross entropy loss
        label_pred = jt.clamp(label_pred, min_v=1e-9, max_v=1.0 - 1e-9)
        label_pred = jt.log(label_pred)
        loss2 = self.criterion2(label_pred, max_id)
        loss2 = jt.mean(loss2 * mask.squeeze(1))
        
        loss = loss1 + self.beta * loss2
        return loss


class BalanceLoss(nn.Module):
    """Balanced Cross Entropy Loss."""
    
    def __init__(self, ignore_index=255, reduction="mean", weight=None):
        super().__init__()
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)

    def execute(self, pred, target):
        """Forward function."""
        prob = nn.softmax(pred, dim=1)
        weighted_pred = nn.log_softmax(pred, dim=1) * (1 - prob) ** 2
        loss = self.criterion(weighted_pred, target)
        return loss


class berHuLoss(nn.Module):
    """berHu Loss for depth estimation."""
    
    def __init__(self, delta=0.2, ignore_index=0, reduction="mean"):
        super().__init__()
        self.delta = delta
        self.ignore_index = ignore_index
        self.reduction = reduction

    def execute(self, pred, target):
        """Forward function."""
        valid_mask = (target != self.ignore_index).float32()
        valid_delta = jt.abs(pred - target) * valid_mask
        max_delta = jt.max(valid_delta)
        delta = self.delta * max_delta

        f_mask = (valid_delta <= delta).float32() * valid_mask
        s_mask = (1 - f_mask) * valid_mask
        f_delta = valid_delta * f_mask
        s_delta = ((valid_delta ** 2) + delta ** 2) / (2 * delta) * s_mask

        loss = jt.mean(f_delta + s_delta)
        return loss


class SigmoidFocalLoss(nn.Module):
    """Sigmoid Focal Loss for binary classification."""
    
    def __init__(self, ignore_label=255, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def execute(self, pred, target):
        """Forward function."""
        b, h, w = target.shape
        pred = pred.view(b, -1, 1)
        pred_sigmoid = nn.sigmoid(pred)
        target = target.view(b, -1).float32()
        mask = (target != self.ignore_label).float32()
        target = mask * target
        onehot = target.view(b, -1, 1)

        max_val = jt.clamp(-pred_sigmoid, min_v=0)

        pos_part = (1 - pred_sigmoid) ** self.gamma * (pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + jt.log(jt.exp(-max_val) + jt.exp(-pred_sigmoid - max_val)))

        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(dim=-1) * mask
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class ProbOhemCrossEntropy2d(nn.Module):
    """Probability-based Online Hard Example Mining Cross Entropy Loss."""
    
    def __init__(self, ignore_label=255, reduction="mean", thresh=0.6, min_kept=256, down_ratio=1, use_weight=False):
        super().__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        
        if use_weight:
            # Default Cityscapes class weights
            weight = jt.array([
                0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                1.0865, 1.1529, 1.0507
            ]).float32()
            self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_label)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_label)

    def execute(self, pred, target):
        """Forward function."""
        b, c, h, w = pred.shape
        target = target.view(-1)
        valid_mask = (target != self.ignore_label)
        target = target * valid_mask.int64()
        num_valid = valid_mask.sum()

        prob = nn.softmax(pred, dim=1)
        prob = prob.permute(1, 0, 2, 3).reshape(c, -1)

        if self.min_kept > num_valid:
            print(f"Labels: {num_valid}")
        elif num_valid > 0:
            prob = prob.masked_fill(~valid_mask, 1.0)
            mask_prob = prob[target, jt.arange(len(target))]
            threshold = self.thresh
            
            if self.min_kept > 0:
                _, indices = jt.argsort(mask_prob)
                threshold_index = indices[min(len(indices), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob <= threshold
                target = target * kept_mask.int64()
                valid_mask = valid_mask & kept_mask

        target = target.masked_fill(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target) 