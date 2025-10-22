import torch
import wandb
from egg.core.callbacks import WandbLogger, Callback
from egg.core.interaction import LoggingStrategy, Interaction
from typing import Any, Dict
import torch

class EarlyStop(Callback):
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.stop = False
        self.epoch = 0
        self.save = False
        self.global_min = float('inf')

    def on_validation_end(self, loss: float, logs, epoch=None):
        self.epoch += 1
        metrics = {key: value.mean().item() for key, value in logs.aux.items()}
        if 'final_symbol_loss' not in metrics:
            return

        current_loss = metrics['final_symbol_loss']
        if self.epoch > self.patience:
            if current_loss < self.global_min - self.min_delta:
                self.global_min = current_loss  # <-- move it here
                self.counter = 0
                self.save = True
            else:
                self.counter += 1
                self.save = False
                if self.counter >= self.patience:
                    self.stop = True

            print(f"[EarlyStop] Epoch={self.epoch} | Loss={current_loss:.4f} | MinLoss={self.global_min:.4f} | Counter={self.counter} | Stop={self.stop}")

class WandbTopSim(WandbLogger):
    def __init__(self, *args, mod=None, epoch=0, **kwargs):
        super(WandbTopSim, self).__init__(*args, **kwargs)
        self.epoch = epoch
        self.mod = mod
    
    def on_train_begin(self, trainer_instance: "Trainer"):  # noqa: F821
        self.trainer = trainer_instance

    @staticmethod
    def safe_log(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        """Logs metrics to wandb, ensuring compatibility with wandb.log."""
        safe_metrics = {}
        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    if value.dim() == 0:  # Convert zero-dimensional tensors to scalars
                        value = value.squeeze().item()
                safe_metrics[key] = value
        if safe_metrics:  # Only log if there are valid metrics
            wandb.log(safe_metrics, commit=commit, **kwargs)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            metrics = {key: value.mean().item() for key, value in logs.aux.items()}
            metrics["epoch"] = self.epoch
            wandb.log(metrics, commit=True)
    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            metrics = {'val_' + key: value.mean().item() for key, value in logs.aux.items()}
            metrics["epoch"] = self.epoch
            wandb.log(metrics, commit=True)
            self.epoch += 1
