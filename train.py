import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.datasets.dataloader import get_dataloaders, get_hybrid_dataloaders
from src.models.crnn_lstm import CRNN
from src.models.dual_branch_crnn import DualBranchCRNN
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Raag Classification Baseline CNN")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-name", type=str, default="baseline_crnn_A", help="Name of the experiment folder")
    parser.add_argument("--feature-type", type=str, default="mel", choices=["mel", "swaragram", "hybrid"], help="Type of input feature")
    args = parser.parse_args()

    # 1. Setup Environment
    set_seed(args.seed)
    
    experiment_dir = Path("experiments") / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(experiment_dir)
    logger.info("=== Starting Training Run ===")
    logger.info(f"Hyperparameters: {vars(args)}")

    # 2. Data Loaders
    logger.info("Initializing DataLoaders...")
    if args.feature_type == "hybrid":
        train_loader, val_loader, test_loader = get_hybrid_dataloaders(batch_size=args.batch_size, num_workers=4)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=4, feature_type=args.feature_type)
    num_classes = train_loader.dataset.num_classes
    label_mapping = train_loader.dataset.raag_to_idx
    
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 3. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.feature_type == "hybrid":
        logger.info(f"Initializing DualBranchCRNN (Hybrid) with {num_classes} classes on {device}")
        model = DualBranchCRNN(num_classes=num_classes)
    else:
        logger.info(f"Initializing CRNN with {num_classes} classes on {device}")
        model = CRNN(input_type=args.feature_type, num_classes=num_classes)
    
    # 4. Optimization
    # Label smoothing (epsilon=0.1) is consistent with the proposal's claim that folk songs
    # may have multiple plausible classical raag associations — we avoid hard one-hot targets.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=experiment_dir,
        patience=args.patience,
        label_mapping=label_mapping
    )

    # 6. Fit
    history = trainer.fit(num_epochs=args.epochs)
    
    logger.info("=== Training Complete ===")
    logger.info(f"Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()
