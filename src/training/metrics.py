import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class MetricTracker:
    """
    Accumulates predictions and targets across an epoch to compute final metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_logits = [] # Store raw logits for Top-k
        self.running_loss = 0.0
        self.num_batches = 0

    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor):
        self.running_loss += loss
        self.num_batches += 1
        
        # Convert logits/probs to class predictions
        if preds.ndim > 1 and preds.size(1) > 1:
            self.all_logits.extend(preds.detach().cpu().numpy())
            preds = torch.argmax(preds, dim=1)
            
        self.all_preds.extend(preds.detach().cpu().numpy())
        self.all_targets.extend(targets.detach().cpu().numpy())

    def compute(self):
        """
        Computes the final metrics for the accumulated epoch data.
        Returns a dictionary of metrics.
        """
        avg_loss = self.running_loss / max(1, self.num_batches)
        
        preds_np = np.array(self.all_preds)
        targets_np = np.array(self.all_targets)
        logits_np = np.array(self.all_logits)
        
        if len(targets_np) == 0:
            return {"loss": avg_loss, "accuracy": 0.0, "top3_accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

        acc = accuracy_score(targets_np, preds_np)
        
        # Compute Top-3 Accuracy if we have logits
        top3_acc = 0.0
        if len(logits_np) > 0 and logits_np.shape[1] >= 3:
            # argsort sorts ascending, so we take the last 3 columns for top 3
            top3_preds = np.argsort(logits_np, axis=1)[:, -3:]
            # Check if target is in the top 3 predictions for each sample
            correct_top3 = [targets_np[i] in top3_preds[i] for i in range(len(targets_np))]
            top3_acc = np.mean(correct_top3)
        else:
            top3_acc = acc # Fallback if logits aren't provided

        prec = precision_score(targets_np, preds_np, average='macro', zero_division=0)
        rec = recall_score(targets_np, preds_np, average='macro', zero_division=0)
        f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "top3_accuracy": top3_acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        }
    
    def get_confusion_matrix(self):
        """
        Returns the raw confusion matrix.
        """
        return confusion_matrix(self.all_targets, self.all_preds)
