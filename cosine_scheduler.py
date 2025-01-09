
import torch.optim as optim
import numpy as np
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters, min_ratio = 0.1):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_ratio = min_ratio
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = (0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters)))*(1-self.min_ratio) + self.min_ratio
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    