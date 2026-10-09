"""
generate_fig5_fixed.py

Directly computes and renders Figure 5: Test Set Normalized Confusion Matrices
Side-by-side comparison:
  - Left Panel: Baseline Log-Mel CRNN (Top-1 Acc: 14.7% — Severe Overfitting / Diffuse Confusions)
  - Right Panel: Proposed Swaragram CRNN (Top-1 Acc: 57.8% — Strong Diagonal / Structured Confusions)

Saved to:
  experiments/paper_figures/fig5_confusion_matrix.png
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Styling
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
})

MEL_COLOR   = "#D90429"   # Crimson Red
SWARA_COLOR = "#0077B6"   # Deep Teal/Blue

def generate_fig5():
    logger.info("Computing confusion matrices from test predictions...")

    # Load label mapping
    mapping_file = PROJECT_ROOT / "data" / "processed" / "metadata" / "label_mapping.json"
    with open(mapping_file) as f:
        mapping = json.load(f)

    # Sorted class names (0 to 37)
    class_names = [k for k, v in sorted(mapping.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    # Load test set prediction CSVs
    mel_csv = PROJECT_ROOT / "experiments" / "baseline_crnn_A" / "song_level_results.csv"
    swara_csv = PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "song_level_results.csv"

    mel_df = pd.read_csv(mel_csv)
    swara_df = pd.read_csv(swara_csv)

    # Compute raw confusion matrices
    cm_mel = confusion_matrix(mel_df["true_idx"], mel_df["pred_idx"], labels=range(num_classes))
    cm_swara = confusion_matrix(swara_df["true_idx"], swara_df["pred_idx"], labels=range(num_classes))

    # Row-normalize (recall per class) with epsilon to avoid div by zero
    cm_mel_norm = cm_mel.astype(float) / (cm_mel.sum(axis=1, keepdims=True) + 1e-8)
    cm_swara_norm = cm_swara.astype(float) / (cm_swara.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))

    # --- Left Panel: Mel CRNN ---
    sns.heatmap(
        cm_mel_norm,
        ax=axes[0],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        cbar_kws={"label": "Normalized Recall (Proportion of Test Recordings)", "shrink": 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.2,
        linecolor="#E5E5E5"
    )
    axes[0].set_title(
        "A: Baseline Log-Mel CRNN\n(Song Top-1 Acc: 14.7% — Diffuse Error Scatter)",
        fontsize=12.5,
        fontweight="bold",
        color="#B7094C",
        pad=12
    )
    axes[0].set_xlabel("Predicted Canonical Raga", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Canonical Raga", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=90, labelsize=8)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8)

    # --- Right Panel: Swaragram CRNN ---
    sns.heatmap(
        cm_swara_norm,
        ax=axes[1],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        cbar_kws={"label": "Normalized Recall (Proportion of Test Recordings)", "shrink": 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.2,
        linecolor="#E5E5E5"
    )
    axes[1].set_title(
        "B: Proposed Swaragram CRNN (Ours)\n(Song Top-1 Acc: 57.8% — Dominant Diagonal / Musically Adjacent Errors)",
        fontsize=12.5,
        fontweight="bold",
        color="#0077B6",
        pad=12
    )
    axes[1].set_xlabel("Predicted Canonical Raga", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Canonical Raga", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=90, labelsize=8)
    axes[1].tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(
        "Figure 5: Test Set Normalized Confusion Matrices (38 Ragas, N = 116 Full Recordings)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = OUT_DIR / "fig5_confusion_matrix.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Successfully generated clean side-by-side Figure 5 at: {out_path}")

if __name__ == "__main__":
    generate_fig5()
