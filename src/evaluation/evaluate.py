import argparse
from pathlib import Path
import json
import logging
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.models.baseline_cnn import BaselineCNN
from src.datasets.dataloader import get_dataloaders
from src.training.metrics import MetricTracker

logger = logging.getLogger("RaagAnalysis")

def evaluate_model(model_path: str, batch_size: int = 32):
    """
    Evaluates a saved model checkpoint on the test set.
    Generates a classification report and a confusion matrix plot.
    """
    model_path = Path(model_path)
    experiment_dir = model_path.parent
    
    # Configure logger for evaluation
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    logger.info(f"Loading checkpoint from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    label_mapping = checkpoint.get('label_mapping', None)
    if label_mapping is None:
        raise ValueError("Checkpoint does not contain 'label_mapping'!")
        
    num_classes = len(label_mapping)
    
    logger.info("Initializing Test DataLoader...")
    _, _, test_loader = get_dataloaders(batch_size=batch_size)
    
    logger.info(f"Initializing BaselineCNN on {device}...")
    model = BaselineCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    tracker = MetricTracker()
    criterion = torch.nn.CrossEntropyLoss()

    logger.info("Running inference on test set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            tracker.update(loss.item(), outputs, targets)

    metrics = tracker.compute()
    
    logger.info("=== Test Set Results ===")
    logger.info(f"Loss:      {metrics['loss']:.4f}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"F1 Macro:  {metrics['f1_macro']:.4f}")
    logger.info(f"Precision: {metrics['precision_macro']:.4f}")
    logger.info(f"Recall:    {metrics['recall_macro']:.4f}")

    # Generate full classification report
    idx_to_raag = {idx: raag for raag, idx in label_mapping.items()}
    class_names = [idx_to_raag[i] for i in range(num_classes)]
    
    all_preds = np.array(tracker.all_preds)
    all_targets = np.array(tracker.all_targets)
    
    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    
    report_file = experiment_dir / "test_report.txt"
    with open(report_file, 'w') as f:
        f.write("=== Test Set Evaluation ===\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n\n")
        f.write(report)
        
    logger.info(f"Saved full report to {report_file}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    cm_file = experiment_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=300)
    logger.info(f"Saved confusion matrix plot to {cm_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Raag Classification Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.batch_size)
