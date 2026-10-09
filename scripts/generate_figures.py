"""
generate_figures.py

Generates all publication-quality figures for a given experiment:
  1. Loss curve (train vs validation)
  2. Accuracy curve with Top-3 (train vs validation)
  3. Validation confusion matrix
  4. Top-3 qualitative prediction table (for Swaragram experiments)
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataloader import get_dataloaders


# ─── Plotting Style ───────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TRAIN_COLOR = "#3A86FF"
VAL_COLOR   = "#FF006E"
TOP3_COLOR  = "#8338EC"


# ─── 1. Learning Curves ───────────────────────────────────────────────────────

def plot_curves(experiment_name):
    print("Generating learning curves...")
    history_path = PROJECT_ROOT / "experiments" / experiment_name / "history.csv"
    if not history_path.exists():
        print(f"  [!] History file not found at {history_path}")
        return

    df = pd.read_csv(history_path)
    out_dir = history_path.parent

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Epoch"], df["Train Loss"], label="Train Loss", color=TRAIN_COLOR, linewidth=2)
    ax.plot(df["Epoch"], df["Val Loss"], label="Validation Loss", color=VAL_COLOR, linewidth=2)
    best_epoch = df.loc[df["Val F1"].astype(float).idxmax(), "Epoch"]
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2, label=f"Best Epoch ({int(best_epoch)})")
    ax.set_title(f"Loss — {experiment_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "loss_curve.png")
    plt.close(fig)
    print("  [OK] loss_curve.png")

    # --- Accuracy + Top-3 curve ---
    has_top3 = "Val Top3" in df.columns and "Train Top3" in df.columns
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Epoch"], df["Train Acc"], label="Train Top-1 Acc", color=TRAIN_COLOR, linewidth=2)
    ax.plot(df["Epoch"], df["Val Acc"], label="Val Top-1 Acc", color=VAL_COLOR, linewidth=2)
    if has_top3:
        ax.plot(df["Epoch"], df["Val Top3"], label="Val Top-3 Acc", color=TOP3_COLOR,
                linewidth=2, linestyle="--")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2, label=f"Best Epoch ({int(best_epoch)})")
    ax.set_title(f"Accuracy -- {experiment_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "accuracy_curve.png")
    plt.close(fig)
    print("  [OK] accuracy_curve.png")


# --- 2. Confusion Matrix ---

def generate_confusion_matrix(experiment_name, feature_type="mel"):
    print("Generating confusion matrix...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = PROJECT_ROOT / "experiments" / experiment_name / "best_model.pth"
    if not model_path.exists():
        print(f"  [!] Model checkpoint not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    label_mapping = checkpoint["label_mapping"]
    num_classes = len(label_mapping)
    class_names = [name for name, _ in sorted(label_mapping.items(), key=lambda x: x[1])]

    if "crnn" in experiment_name.lower() or "swaragram" in experiment_name.lower():
        from src.models.crnn_lstm import CRNN
        model = CRNN(input_type=feature_type, num_classes=num_classes)
    else:
        from src.models.baseline_cnn import BaselineCNN
        model = BaselineCNN(num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, val_loader, _ = get_dataloaders(batch_size=64, num_workers=4, feature_type=feature_type)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation Inference"):
            inputs = inputs.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(inputs)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.numpy())

    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cm_norm, annot=False, cmap="Blues", fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Validation Confusion Matrix (Normalized) -- {experiment_name}",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("True Raag", fontsize=12)
    ax.set_xlabel("Predicted Raag", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.savefig(PROJECT_ROOT / "experiments" / experiment_name / "confusion_matrix.png")
    plt.close(fig)
    print("  [OK] confusion_matrix.png")


# --- 3. Top-3 Qualitative Table ---

def generate_top3_table(experiment_name, feature_type="swaragram", n_samples=10):
    print("Generating Top-3 qualitative table...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = PROJECT_ROOT / "experiments" / experiment_name / "best_model.pth"
    if not model_path.exists():
        print("  [!] Model checkpoint not found.")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    label_mapping = checkpoint["label_mapping"]
    num_classes = len(label_mapping)
    idx_to_raag = {v: k for k, v in label_mapping.items()}

    from src.models.crnn_lstm import CRNN
    model = CRNN(input_type=feature_type, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=1, num_workers=0, feature_type=feature_type)

    rows = []
    seen_labels = set()

    with torch.no_grad():
        for inputs, targets in test_loader:
            true_label = targets[0].item()
            if true_label in seen_labels:
                continue
            seen_labels.add(true_label)

            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

            top3_idx = np.argsort(probs)[::-1][:3]
            top3_raags = [idx_to_raag[i] for i in top3_idx]
            top3_probs = [probs[i] for i in top3_idx]

            true_raag = idx_to_raag[true_label]
            correct = true_label == top3_idx[0]
            in_top3 = true_label in top3_idx

            rows.append({
                "True Raag": true_raag,
                "Pred 1 (prob)": f"{top3_raags[0]} ({top3_probs[0]:.2f})",
                "Pred 2 (prob)": f"{top3_raags[1]} ({top3_probs[1]:.2f})",
                "Pred 3 (prob)": f"{top3_raags[2]} ({top3_probs[2]:.2f})",
                "Top-1": "[OK]" if correct else "[X]",
                "Top-3": "[OK]" if in_top3 else "[X]",
            })

            if len(rows) >= n_samples:
                break

    table_df = pd.DataFrame(rows)
    csv_path = PROJECT_ROOT / "experiments" / experiment_name / "top3_qualitative_table.csv"
    table_df.to_csv(csv_path, index=False)
    print("  [OK] top3_qualitative_table.csv")
    print()
    print(table_df.to_string(index=False))


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--feature-type", type=str, default="mel",
                        choices=["mel", "swaragram"],
                        help="Feature type used in the experiment")
    parser.add_argument("--skip-cm", action="store_true", help="Skip confusion matrix (slower)")
    parser.add_argument("--top3-table", action="store_true",
                        help="Generate qualitative Top-3 prediction table")
    args = parser.parse_args()

    plot_curves(args.experiment)
    
    if not args.skip_cm:
        generate_confusion_matrix(args.experiment, feature_type=args.feature_type)
    
    if args.top3_table:
        generate_top3_table(args.experiment, feature_type=args.feature_type)

    print(f"\nAll figures saved to experiments/{args.experiment}/")
