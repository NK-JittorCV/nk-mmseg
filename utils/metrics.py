"""
Evaluation metrics for semantic segmentation tasks.
All implementations are pure Jittor without PyTorch dependencies.
"""

import numpy as np
import jittor as jt


class SegmentationMetric:
    """Segmentation metric calculator."""
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset the metric."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """Update the metric with new predictions and targets."""
        if isinstance(pred, jt.Var):
            pred = pred.numpy()
        if isinstance(target, jt.Var):
            target = target.numpy()
        
        # Flatten the arrays
        pred = pred.flatten()
        target = target.flatten()
        
        # Remove ignored pixels
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # Compute confusion matrix
        n = self.num_classes
        for lt, lp in zip(target, pred):
            if lt < n and lp < n:
                self.confusion_matrix[lt][lp] += 1
    
    def get_confusion_matrix(self):
        """Get the confusion matrix."""
        return self.confusion_matrix
    
    def get_pixel_acc(self):
        """Get pixel accuracy."""
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc
    
    def get_mean_acc(self):
        """Get mean class accuracy."""
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc
    
    def get_iou(self):
        """Get IoU for each class."""
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / (union + 1e-8)
        return iou
    
    def get_miou(self):
        """Get mean IoU."""
        iou = self.get_iou()
        miou = np.nanmean(iou)
        return miou
    
    def get_freq_weighted_iou(self):
        """Get frequency weighted IoU."""
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iou = self.get_iou()
        freq_weighted_iou = np.sum(freq * iou)
        return freq_weighted_iou
    
    def get_results(self):
        """Get all metrics as a dictionary."""
        results = {
            'pixel_acc': self.get_pixel_acc(),
            'mean_acc': self.get_mean_acc(),
            'miou': self.get_miou(),
            'freq_weighted_iou': self.get_freq_weighted_iou(),
            'iou_per_class': self.get_iou()
        }
        return results


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """Update the meter with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Progress meter for training/validation."""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        """Display progress."""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        """Get batch format string."""
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def intersect_and_union(pred_label, label, num_classes, ignore_index=255):
    """Calculate intersection and union for semantic segmentation."""
    if isinstance(pred_label, jt.Var):
        pred_label = pred_label.numpy()
    if isinstance(label, jt.Var):
        label = label.numpy()
    
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    
    intersect = pred_label[pred_label == label]
    area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes - 1))[0]
    area_pred_label = np.histogram(pred_label, bins=num_classes, range=(0, num_classes - 1))[0]
    area_label = np.histogram(label, bins=num_classes, range=(0, num_classes - 1))[0]
    area_union = area_pred_label + area_label - area_intersect
    
    return area_intersect, area_union, area_pred_label, area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index=255):
    """Calculate mean IoU."""
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    
    total_area_intersect = np.zeros((num_classes,), dtype=np.float64)
    total_area_union = np.zeros((num_classes,), dtype=np.float64)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float64)
    total_area_label = np.zeros((num_classes,), dtype=np.float64)
    
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            results[i], gt_seg_maps[i], num_classes, ignore_index
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union
    
    return all_acc, acc, iou 