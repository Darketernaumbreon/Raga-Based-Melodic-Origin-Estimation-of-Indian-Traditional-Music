"""
evaluate_song_level.py

Song-Level Aggregation Evaluation (Option 7)
============================================
Instead of classifying each 30-second segment independently, this script:

  1. Groups all test segments by recording_id.
  2. Runs every segment through the trained CRNN to get a probability distribution.
  3. Averages the per-segment probability distributions across the whole recording
     (mean-pooling in probability space == implicit majority vote + confidence weighting).
  4. Makes a single Top-1 and Top-3 prediction per recording.
  5. Computes and reports song-level Accuracy, Top-3 Accuracy, and Macro-F1.

This is the most principled change to improve performance without retraining --
random segment-level errors cancel out, and confident correct predictions dominate.

Usage:
    python scripts/evaluate_song_level.py --experiment swaragram_crnn_A --feature-type swaragram
    python scripts/evaluate_song_level.py --experiment baseline_crnn_A  --feature-type mel
"""
import sys
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class SegmentDataset(torch.utils.data.Dataset):
    """
    Loads segments from ONE split (test by default), grouped by recording.
    Returns (feature_tensor, label_idx, recording_id) per segment.
    """
    def __init__(self, split, feature_type, features_root, label_mapping,
                 stats_path=None):
        self.feature_type = feature_type
        self.features_root = Path(features_root)
        self.label_mapping = label_mapping

        meta_csv = {
            "mel": self.features_root / "mel_segments.csv",
            "swaragram": self.features_root / "swaragram_segments.csv",
        }[feature_type]

        df = pd.read_csv(meta_csv)
        self.df = df[df["feature_file"].str.contains(split, regex=False)].reset_index(drop=True)
        logger.info(f"  {split}: {len(self.df)} segments | "
                    f"{self.df['recording_id'].nunique()} recordings")

        # Mel z-score stats
        self.train_mean = self.train_std = None
        if feature_type == "mel" and stats_path and Path(stats_path).exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.train_mean = stats["train_mean"]
            self.train_std  = stats["train_std"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = row["feature_file"].replace("\\", "/")
        feat = np.load(self.features_root / rel).astype(np.float32)

        if self.feature_type == "mel" and self.train_mean is not None:
            feat = (feat - self.train_mean) / (self.train_std + 1e-8)

        tensor = torch.FloatTensor(feat).unsqueeze(0)   # (1, bins, T)
        label_idx = self.label_mapping[row["label"]]
        return tensor, label_idx, str(row["recording_id"])


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _remap_legacy_state_dict(state_dict):
    """
    Remap old flat-key checkpoints to the current shared_encoder / melodic_origin_head layout.

    Old layout:          New layout:
      backbone.X      ->   shared_encoder.backbone.X
      encoder.X       ->   shared_encoder.encoder.X
      attention.X     ->   shared_encoder.attention.X
      classifier.X    ->   melodic_origin_head.X
    """
    is_legacy = any(k.startswith("backbone.") for k in state_dict)
    if not is_legacy:
        return state_dict

    logger.info("  Legacy checkpoint detected — remapping state_dict keys...")
    new_sd = {}
    prefix_map = {
        "backbone.":   "shared_encoder.backbone.",
        "encoder.":    "shared_encoder.encoder.",
        "attention.":  "shared_encoder.attention.",
        "classifier.": "melodic_origin_head.",
    }
    for k, v in state_dict.items():
        new_k = k
        for old_pfx, new_pfx in prefix_map.items():
            if k.startswith(old_pfx):
                new_k = new_pfx + k[len(old_pfx):]
                break
        new_sd[new_k] = v
    return new_sd


def load_model(experiment_name, feature_type, device):
    model_path = PROJECT_ROOT / "experiments" / experiment_name / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    label_mapping = checkpoint["label_mapping"]
    num_classes = len(label_mapping)

    if "crnn" in experiment_name.lower() or "swaragram" in experiment_name.lower():
        from src.models.crnn_lstm import CRNN
        model = CRNN(input_type=feature_type, num_classes=num_classes)
    else:
        from src.models.baseline_cnn import BaselineCNN
        model = BaselineCNN(num_classes=num_classes)

    state_dict = _remap_legacy_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, label_mapping


def run_song_level_evaluation(experiment_name, feature_type, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model & label mapping
    model, label_mapping = load_model(experiment_name, feature_type, device)
    idx_to_raag = {v: k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)

    features_root = PROJECT_ROOT / "data" / "processed" / "features"
    stats_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "dataset_stats.json"

    dataset = SegmentDataset(
        split=split,
        feature_type=feature_type,
        features_root=features_root,
        label_mapping=label_mapping,
        stats_path=stats_path,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False
    )

    # ---- Segment-level inference ----
    # Accumulate probability sums per recording_id
    rec_prob_sum  = defaultdict(lambda: np.zeros(num_classes, dtype=np.float64))
    rec_seg_count = defaultdict(int)
    rec_true_label = {}   # recording_id -> ground-truth label index

    logger.info("Running segment-level inference...")
    with torch.no_grad():
        for tensors, label_idxs, rec_ids in tqdm(loader, desc="Inference"):
            tensors = tensors.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()   # (B, C)

            for prob_vec, lbl_idx, rec_id in zip(probs, label_idxs.numpy(), rec_ids):
                rec_prob_sum[rec_id]  += prob_vec
                rec_seg_count[rec_id] += 1
                if rec_id not in rec_true_label:
                    rec_true_label[rec_id] = int(lbl_idx)

    # ---- Song-level aggregation (mean pooling) ----
    results = []
    for rec_id, true_idx in rec_true_label.items():
        avg_prob  = rec_prob_sum[rec_id] / rec_seg_count[rec_id]
        top3_idx  = np.argsort(avg_prob)[::-1][:3]

        pred_idx  = top3_idx[0]
        in_top3   = true_idx in top3_idx

        results.append({
            "recording_id"    : rec_id,
            "true_label"      : idx_to_raag[true_idx],
            "pred_label"      : idx_to_raag[pred_idx],
            "top3_labels"     : [idx_to_raag[i] for i in top3_idx],
            "top3_probs"      : avg_prob[top3_idx].tolist(),
            "n_segments"      : rec_seg_count[rec_id],
            "top1_correct"    : pred_idx == true_idx,
            "top3_correct"    : in_top3,
            "true_idx"        : true_idx,
            "pred_idx"        : pred_idx,
        })

    df = pd.DataFrame(results)

    # ---- Metrics ----
    top1_acc = accuracy_score(df["true_idx"], df["pred_idx"])
    top3_acc = df["top3_correct"].mean()
    macro_f1 = f1_score(df["true_idx"], df["pred_idx"], average="macro", zero_division=0)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  SONG-LEVEL RESULTS  ({experiment_name}  |  {split})")
    logger.info("=" * 60)
    logger.info(f"  Recordings evaluated : {len(df)}")
    logger.info(f"  Top-1 Accuracy       : {top1_acc:.4f}  ({top1_acc*100:.1f}%)")
    logger.info(f"  Top-3 Accuracy       : {top3_acc:.4f}  ({top3_acc*100:.1f}%)")
    logger.info(f"  Macro F1             : {macro_f1:.4f}")
    logger.info("=" * 60)

    # ---- Save results ----
    out_dir = PROJECT_ROOT / "experiments" / experiment_name
    df.to_csv(out_dir / "song_level_results.csv", index=False)
    logger.info(f"Per-recording results saved to {out_dir}/song_level_results.csv")

    # ---- Per-class breakdown ----
    report = classification_report(
        df["true_label"], df["pred_label"],
        output_dict=False, zero_division=0
    )
    report_path = out_dir / "song_level_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Classification report saved to {out_dir}/song_level_classification_report.txt")

    # ---- Song-level Confusion Matrix ----
    class_names = [idx_to_raag[i] for i in range(num_classes)]
    cm = confusion_matrix(df["true_idx"], df["pred_idx"], labels=range(num_classes))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cm_norm, annot=False, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Song-Level Confusion Matrix -- {experiment_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Raag", fontsize=12)
    ax.set_xlabel("Predicted Raag", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.savefig(out_dir / "song_level_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Song-level confusion matrix saved to {out_dir}/song_level_confusion_matrix.png")

    # ---- Sample qualitative table ----
    logger.info("")
    logger.info("Sample per-recording predictions:")
    sample = df.sample(min(12, len(df)), random_state=42)
    for _, row in sample.iterrows():
        mark = "[OK]" if row["top1_correct"] else ("[T3]" if row["top3_correct"] else "[X]")
        top3_str = ", ".join(
            f"{r} ({p:.2f})" for r, p in zip(row["top3_labels"], row["top3_probs"])
        )
        logger.info(f"  {mark}  True: {row['true_label']:20s}  |  Top-3: {top3_str}  "
                    f"  (n={row['n_segments']} segs)")

    return {
        "top1_acc" : top1_acc,
        "top3_acc" : top3_acc,
        "macro_f1" : macro_f1,
        "n_recs"   : len(df),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",   type=str, required=True,
                        help="Experiment folder name, e.g. swaragram_crnn_A")
    parser.add_argument("--feature-type", type=str, default="swaragram",
                        choices=["mel", "swaragram"])
    parser.add_argument("--split",        type=str, default="test",
                        choices=["test", "validation"])
    args = parser.parse_args()

    logger.info(f"=== Song-Level Aggregation Evaluation ===")
    logger.info(f"Experiment  : {args.experiment}")
    logger.info(f"Feature     : {args.feature_type}")
    logger.info(f"Split       : {args.split}")

    run_song_level_evaluation(
        experiment_name=args.experiment,
        feature_type=args.feature_type,
        split=args.split,
    )
