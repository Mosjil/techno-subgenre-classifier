import torch

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
