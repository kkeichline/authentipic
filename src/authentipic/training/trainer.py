# src/authentipic/training/trainer.py
import os
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import logging
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logging.getLogger(__name__)

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "min" and score > self.best_score - self.min_delta:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == "max" and score < self.best_score + self.min_delta:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str,
        scheduler: Optional[_LRScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save a checkpoint of the model."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save the last checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "last_checkpoint.pth")

        # Save the best model if it's the best so far
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            self.logger.info(
                f"Saved new best model with validation loss: {val_loss:.4f}"
            )

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint and return the epoch to resume from."""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(
                f"Checkpoint {checkpoint_path} does not exist. Starting from scratch."
            )
            return 0

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}"
        )
        return checkpoint["epoch"]

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(dataloader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / total
        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = total_loss / len(dataloader)
        val_acc = correct / total
        return {"loss": val_loss, "accuracy": val_acc}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        resume_from: Optional[str] = None,
    ):
        start_epoch = 0
        best_val_loss = float("inf")

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")

            train_metrics = self.train_epoch(train_loader)
            self.logger.info(
                f"Training Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}"
            )

            val_metrics = self.validate(val_loader)
            self.logger.info(
                f"Validation Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}"
            )

            # Save checkpoint
            is_best = val_metrics["loss"] < best_val_loss
            best_val_loss = min(val_metrics["loss"], best_val_loss)
            self.save_checkpoint(epoch, val_metrics["loss"], is_best)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

                self.logger.info(
                    f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            if self.early_stopping is not None:
                self.early_stopping(val_metrics["loss"])
                if self.early_stopping.early_stop:
                    self.logger.info("Early stopping")
                    break

        self.logger.info("Training completed")

    @staticmethod
    def create_scheduler(
        scheduler_type: str, optimizer: torch.optim.Optimizer, **kwargs
    ) -> _LRScheduler:
        if scheduler_type == "step":
            return StepLR(optimizer, **kwargs)
        elif scheduler_type == "exponential":
            return ExponentialLR(optimizer, **kwargs)
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(optimizer, **kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    @staticmethod
    def create_early_stopping(
        patience: int = 7, min_delta: float = 0, mode: str = "min"
    ) -> EarlyStopping:
        return EarlyStopping(patience=patience, min_delta=min_delta, mode=mode)
