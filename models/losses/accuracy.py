"""
Accuracy metric module converted to Jittor.
"""

import jittor as jt
from jittor import nn

__all__ = ['accuracy', 'Accuracy']


def accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate top-k accuracy.
    Args:
        pred (jt.Var): shape (N, C, ...)
        target (jt.Var): shape (N, ...)
    Returns:
        float | tuple: accuracy values
    """
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    N = pred.shape[0]
    pred_flat = pred.reshape(N, pred.shape[1], -1)
    target_flat = target.reshape(N, -1)

    num_el = pred_flat.shape[2]
    pred_flat = pred_flat.permute(0, 2, 1).reshape(-1, pred.shape[1])  # (N*L, C)
    target_flat = target_flat.reshape(-1)  # (N*L)

    if ignore_index is not None:
        valid_mask = (target_flat != ignore_index)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

    if pred_flat.numel() == 0:
        acc = [0.0 for _ in topk]
        return acc[0] if return_single else acc

    vals, inds = jt.topk(pred_flat, k=maxk, dim=1)

    correct = (inds == target_flat.unsqueeze(1)).float32()
    if thresh is not None:
        correct = correct * (vals > thresh).float32()

    res = []
    eps = 1e-12
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(float(correct_k) * 100.0 / (target_flat.numel() + eps))

    return res[0] if return_single else tuple(res)


class Accuracy(nn.Module):
    def __init__(self, topk=(1,), thresh=None, ignore_index=None):
        super().__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = ignore_index

    def execute(self, pred, target):
        return accuracy(pred, target, self.topk, self.thresh, self.ignore_index) 