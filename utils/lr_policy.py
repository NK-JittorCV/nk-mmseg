"""
Learning rate policy for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import math
import jittor as jt


class WarmUpPolyLR(object):
    """Warm up polynomial learning rate scheduler."""
    
    def __init__(self, optimizer, target_lr=0, power=0.9, max_iters=0, warmup_factor=1.0/3, warmup_iters=500, warmup_method='linear', last_epoch=-1):
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.power = power
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.last_epoch = last_epoch
        
        # Handle different optimizer structures
        if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
            self.base_lrs = [group.get('lr', optimizer.lr) for group in optimizer.param_groups]
        else:
            self.base_lrs = [optimizer.lr]

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError(f"Unknown warmup method: {self.warmup_method}")
            
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            factor = (1 - (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters)) ** self.power
            return [base_lr * factor for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        lrs = self.get_lr()
        
        if hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
        else:
            self.optimizer.lr = lrs[0]


class PolyLR(object):
    """Polynomial learning rate scheduler."""
    
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.power = power
        self.last_epoch = last_epoch
        
        if not isinstance(optimizer.param_groups, list):
            self.base_lrs = [optimizer.lr]
        else:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        factor = (1 - self.last_epoch / self.max_iters) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        lrs = self.get_lr()
        
        if hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lrs[0]


class CosineAnnealingLR(object):
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        
        if not isinstance(optimizer.param_groups, list):
            self.base_lrs = [optimizer.lr]
        else:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        lrs = self.get_lr()
        
        if hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lrs[0]


class StepLR(object):
    """Step learning rate scheduler."""
    
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        
        if not isinstance(optimizer.param_groups, list):
            self.base_lrs = [optimizer.lr]
        else:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        lrs = self.get_lr()
        
        if hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lrs[0]


class MultiStepLR(object):
    """Multi-step learning rate scheduler."""
    
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.milestones = sorted(list(milestones))
        self.gamma = gamma
        self.last_epoch = last_epoch
        
        if not isinstance(optimizer.param_groups, list):
            self.base_lrs = [optimizer.lr]
        else:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        return [base_lr * self.gamma ** self.get_milestone_count()
                for base_lr in self.base_lrs]

    def get_milestone_count(self):
        """Get number of milestones passed."""
        return sum([1 for milestone in self.milestones if milestone <= self.last_epoch])

    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        lrs = self.get_lr()
        
        if hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lrs[0]
