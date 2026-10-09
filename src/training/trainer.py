import logging
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import MetricTracker
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import save_checkpoint

logger = logging.getLogger("RaagAnalysis")

class Trainer:
    """
    PyTorch Model Trainer.
    Encapsulates the training cycle across epochs, validation checks, and saving the best model.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        patience: int = 10,
        label_mapping: dict = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        self.scheduler = scheduler
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.label_mapping = label_mapping

        # Metrics & Early Stopping
        self.train_tracker = MetricTracker()
        self.val_tracker = MetricTracker()
        
        # Maximize Validation F1 macro
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda')
        self.best_val_score = -float('inf')

        logger.info(f"Trainer initialized on device: {self.device}")
        
        # Initialize history CSV
        self.history_file = self.checkpoint_dir / "history.csv"
        self._init_history_file()

    def _init_history_file(self):
        with open(self.history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", 
                "Train Loss", "Train Acc", "Train Top3", "Train F1",
                "Val Loss", "Val Acc", "Val Top3", "Val Precision", "Val Recall", "Val F1",
                "Learning Rate"
            ])

    def _log_history(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float):
        with open(self.history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_metrics['loss']:.4f}", f"{train_metrics['accuracy']:.4f}", f"{train_metrics['top3_accuracy']:.4f}", f"{train_metrics['f1_macro']:.4f}",
                f"{val_metrics['loss']:.4f}", f"{val_metrics['accuracy']:.4f}", f"{val_metrics['top3_accuracy']:.4f}",
                f"{val_metrics['precision_macro']:.4f}", f"{val_metrics['recall_macro']:.4f}", f"{val_metrics['f1_macro']:.4f}",
                f"{lr:.2e}"
            ])

    def train_epoch(self) -> dict:
        self.model.train()
        self.train_tracker.reset()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for i, (inputs, targets) in enumerate(pbar):
            # Handle hybrid batches: inputs = (mel_tensor, swara_tensor)
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(self.device) for x in inputs)
            else:
                inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                outputs = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.train_tracker.update(loss.item(), outputs, targets)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return self.train_tracker.compute()

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        self.val_tracker.reset()

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for i, (inputs, targets) in enumerate(pbar):
            # Handle hybrid batches: inputs = (mel_tensor, swara_tensor)
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(self.device) for x in inputs)
            else:
                inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                outputs = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.val_tracker.update(loss.item(), outputs, targets)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return self.val_tracker.compute()

    def fit(self, num_epochs: int = 50) -> Dict[str, list]:
        logger.info(f"Starting training loop for {num_epochs} epochs...")
        history = []

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # 1. Train
            train_metrics = self.train_epoch()
            
            # 2. Validate
            val_metrics = self.validate()
            
            # 3. Learning Rate Scheduler (ReduceLROnPlateau usually steps on Val Loss)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                    
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 4. Save History
            self._log_history(epoch, train_metrics, val_metrics, current_lr)
            history.append({"train": train_metrics, "val": val_metrics})

            # 5. Logging
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch}/{num_epochs} [{epoch_time:.1f}s]")
            logger.info(f"  Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | Top3: {train_metrics['top3_accuracy']:.4f} | F1: {train_metrics['f1_macro']:.4f}")
            logger.info(f"  Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | Top3: {val_metrics['top3_accuracy']:.4f} | F1: {val_metrics['f1_macro']:.4f}")
            
            # 6. Checkpointing
            # We track Validation Macro F1 as our primary metric for saving the best model
            val_score = val_metrics['f1_macro']
            
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                logger.info(f"  *** New Best Model! Val F1: {val_score:.4f} ***")
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics, 
                    self.checkpoint_dir / "best_model.pth", self.label_mapping
                )
                
            # Always save the last model
            save_checkpoint(
                self.model, self.optimizer, epoch, val_metrics, 
                self.checkpoint_dir / "last_model.pth", self.label_mapping
            )

            # 7. Early Stopping
            if self.early_stopping(val_score):
                logger.info(f"Early stopping triggered! Training halted at epoch {epoch}.")
                break

        return history
