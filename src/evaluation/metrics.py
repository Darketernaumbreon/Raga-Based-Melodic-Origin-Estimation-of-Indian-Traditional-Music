"""
Evaluation and Metrics Utilities for Raag Classification.

This file is intentionally scaffolded with TODOs so you can practice computing Top-K accuracy,
generating confusion matrices, and running evaluation passes on the test split.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 3, 5)) -> Dict[str, float]:
    """
    Compute Top-K accuracy across a batch or dataset of predictions.

    Parameters
    ----------
    outputs : torch.Tensor
        Model logit predictions of shape (Batch_Size, Num_Classes)
    targets : torch.Tensor
        True ground-truth integer labels of shape (Batch_Size,)
    topk : Tuple[int, ...], optional
        Tuple of k values to evaluate (e.g., top-1, top-3, top-5), by default (1, 3, 5)

    Returns
    -------
    Dict[str, float]
        Dictionary mapping top-k names (e.g., 'top_1_acc') to accuracy percentages.
    """
    # TODO: Use torch.topk to find the top k predicted class indices for each sample.
    # Check if the true target label is within the top k predicted indices.
    # Calculate accuracy percentage for each k.
    raise NotImplementedError("Implement compute_topk_accuracy()")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a trained model on the test split.

    Returns
    -------
    Dict containing:
        - 'loss': Average test loss
        - 'top_1_acc': Top-1 accuracy percentage
        - 'top_3_acc': Top-3 accuracy percentage
        - 'y_true': NumPy array of true ground-truth labels
        - 'y_pred': NumPy array of predicted class labels
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    model.eval()
    model.to(device_obj)

    # TODO 1: Loop through all batches in test_loader without tracking gradients (@torch.no_grad())
    # TODO 2: Collect all model outputs and true targets across the entire test set
    # TODO 3: Compute average loss and top-k accuracies using `compute_topk_accuracy()`
    # TODO 4: Return dictionary of results along with `y_true` and `y_pred` for classification reporting
    raise NotImplementedError("Implement evaluate_model()")


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, idx_to_label: Optional[Dict[int, str]] = None) -> str:
    """
    Generate a formatted text report showing precision, recall, and F1-score per raag class.
    You can use `sklearn.metrics.classification_report` here!
    """
    from sklearn.metrics import classification_report
    # TODO: Pass y_true and y_pred into classification_report with target_names
    return ""
