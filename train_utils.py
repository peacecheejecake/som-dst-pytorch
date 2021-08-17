import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingAfterWarmUpScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        cycle_steps: int,
        max_lr: float,
        min_lr: float,
        damping_ratio: float,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.damping_ratio = damping_ratio
        super(CosineAnnealingAfterWarmUpScheduler, self).__init__(optimizer=optimizer, verbose=verbose)


    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) / self.warmup_steps * self._step_count
        else:
            if self.cycle_steps > self.warmup_steps:
                x = (self._step_count - self.warmup_steps) / (self.cycle_steps - self.warmup_steps) / 2 * math.pi
            else:
                x = (self._step_count - self.warmup_steps) / 2 * math.pi
            return self.min_lr + (self.max_lr - self.min_lr) * math.cos(x)


    def step(self):
        lr = self.get_lr()
        # for i, param_group in enumerate(self.optimizer.param_groups):
        #     param_group['lr'] = lr
            # self.print_lr(self.verbose, i, lr)
        self.optimizer.param_groups[0]['lr'] = lr
        self._step_count += 1



def convert_millisecond_to_str(millisecond: float, include_decimal=False):
    r'''Convert time to str
    Args:
        millisecond (float): elapsed time recorded by torch.cuda.Event
        include_decimal (bool): whether include decimal points to second
    '''
    second, decimal = divmod(int(millisecond), 1000)
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    decimal = str(decimal).rjust(3, '0')

    time_str = f'{minute:02d}:{second:02d}'
    if hour > 0:
        time_str = f'{hour:02d}:' + time_str

    if include_decimal:
        time_str += f'.{decimal}'
    
    return time_str
