import torch.nn as nn
from datetime import datetime
from modules import BatchNorm2d_stats
from torch.optim.lr_scheduler import _LRScheduler

def current_time():
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y, %H:%M:%S")
    return date_time

def modify_modules(module,att_stack):
    if isinstance(module, nn.BatchNorm2d):
        return BatchNorm2d_stats(module.num_features)
    else:
        module_son = modify_modules(getattr(module,att_stack[0]),att_stack[1:])
        setattr(module, att_stack[0], module_son)
        return module
    
class Poly(_LRScheduler):

    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=10, last_epoch=-1):

        self.iters_per_epoch = iters_per_epoch

        self.cur_iter = 0

        self.N = num_epochs * iters_per_epoch

        self.warmup_iters = warmup_epochs * iters_per_epoch

        super(Poly, self).__init__(optimizer, last_epoch)



    def get_lr(self):

        T = self.last_epoch * self.iters_per_epoch + self.cur_iter

        factor =  pow((1 - 1.0 * T / self.N), 0.9)

        if self.warmup_iters > 0 and T < self.warmup_iters:

            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch

        self.cur_iter += 1

        assert factor >= 0, 'error in lr_scheduler'

        return [base_lr * factor for base_lr in self.base_lrs]
