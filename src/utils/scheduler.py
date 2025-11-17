import torch
from transformers import get_cosine_schedule_with_warmup

class LRSchedulerWrapper:
    def __init__(self, optimizer, mode='max', factor=0.5, patience=3, threshold=1e-3):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
        )

    def step(self, metric):
        self.scheduler.step(metric)

    def get_lr(self):
        return self.scheduler.optimizer.param_groups[0]['lr']


class WarmupCosineScheduler:

    def __init__(self, optimizer, num_warmup_steps, num_training_steps):
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.optimizer = optimizer

    def step(self):
        self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
