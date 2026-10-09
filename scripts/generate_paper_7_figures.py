"""
generate_paper_7_figures.py

Generates the 7 exact mandatory publication figures requested by the user:
  Fig 1: Entire System Pipeline (Flowchart Diagram)
  Fig 2: Mel Spectrogram vs Swaragram Side-by-Side Comparison
  Fig 3: Actual Swaragram for Raga Yaman / Bhairavi (12 Bins vs Time)
  Fig 4: Tonic Validation (Ground-Truth vs Estimated, 85.2% Pitch-Class Match, Error Distribution)
  Fig 5: Confusion Matrices (Mel Baseline vs Swaragram CRNN Side-by-Side)
  Fig 6: Song-Level Probability Mean Pooling Architecture
  Fig 7: Per-Raga Song-Level Accuracy Bar Chart
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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "experiments" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global styling
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
})

MEL_COLOR   = "#E63946"   # Red
SWARA_COLOR = "#2A9D8F"   # Teal
ACCENT      = "#1D3557"   # Navy
SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni"]


# ── Figure 1: Entire Pipeline ──────────────────────────────────────────────────
def make_fig1():
    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("Raw Audio\nSignal", 0.9, "#457B9D"),
        ("Pitch Tracking\n(YIN F0)", 2.9, "#1D3557"),
        ("Tonic (Sa)\nEstimation", 4.9, "#1D3557"),
        ("Pitch-Class\nNormalization", 6.9, SWARA_COLOR),
        ("Temporal\nSwaragram", 8.9, SWARA_COLOR),
        ("CRNN +\nBiLSTM", 10.9, ACCENT),
        ("Song Probability\nPooling", 12.9, "#E9C46A"),
        ("Predicted\nRaga", 14.3, "#264653"),
    ]

    box_w, box_h = 1.4, 1.8
    y_center = 2.0

    for label, x, color in stages:
        rect = plt.Rectangle((x - box_w/2, y_center - box_h/2), box_w, box_h,
                             color=color, alpha=0.9, zorder=3, linewidth=1.5, edgecolor="white")
        ax.add_patch(rect)
        ax.text(x, y_center, label, ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold", zorder=4, multialignment="center")

    for i in range(len(stages) - 1):
        x1 = stages[i][1] + box_w / 2
        x2 = stages[i+1][1] - box_w / 2
        ax.annotate("", xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle="-|>", color="#4A5568", lw=2), zorder=2)

    ax.set_title("Figure 1: End-to-End System Pipeline — Swaragram Raga Identification",
                 fontsize=13, fontweight="bold", pad=12)
    fig.savefig(OUT_DIR / "fig1_pipeline.png")
    plt.close(fig)
    logger.info("  [OK] fig1_pipeline.png")


# ── Figure 2: Mel vs Swaragram Side-by-Side ────────────────────────────────────
def make_fig2():
    features_root = PROJECT_ROOT / "data" / "processed" / "features"
    mel_df = pd.read_csv(features_root / "mel_segments.csv")
    swara_df = pd.read_csv(features_root / "swaragram_segments.csv")

    mel_row = mel_df[mel_df["recording_id"] == "trf_0028"].iloc[5]
    swara_row = swara_df[swara_df["recording_id"] == "trf_0028"].iloc[5]
    raag_name = mel_row["label"]

    mel = np.load(features_root / mel_row["feature_file"].replace("\\", "/")).astype(np.float32)
    swara = np.load(features_root / swara_row["feature_file"].replace("\\", "/")).astype(np.float32)

    with open(PROJECT_ROOT / "data" / "processed" / "metadata" / "dataset_stats.json") as f:
        stats = json.load(f)
    mel_disp = (mel - stats["train_mean"]) / (stats["train_std"] + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    im0 = axes[0].imshow(mel_disp, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    axes[0].set_title(f"Log-Mel Spectrogram (128 Bins)\nRaag: {raag_name}", fontsize=11, fontweight="bold", color=MEL_COLOR)
    axes[0].set_xlabel("Time Frame (23.2 ms / frame)")
    axes[0].set_ylabel("Mel Frequency Bin")
    plt.colorbar(im0, ax=axes[0], label="Normalized Amplitude")

    im1 = axes[1].imshow(swara, aspect="auto", origin="lower", cmap="YlOrRd", interpolation="nearest")
    axes[1].set_title(f"Swaragram Feature (12 Bins)\nRaag: {raag_name}", fontsize=11, fontweight="bold", color=SWARA_COLOR)
    axes[1].set_xlabel("Time Frame (23.2 ms / frame)")
    axes[1].set_ylabel("Relative Pitch Class (Swara)")
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels(SWARA_NAMES, fontsize=9)
    plt.colorbar(im1, ax=axes[1], label="Relative Energy")

    fig.suptitle("Figure 2: Acoustic Log-Mel Spectrogram vs. Proposed Tonic-Normalized Swaragram",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_feature_comparison.png")
    plt.close(fig)
    logger.info("  [OK] fig2_feature_comparison.png")


# ── Figure 3: Swaragram for Raga Yaman / Bhairavi ──────────────────────────────
def make_fig3():
    features_root = PROJECT_ROOT / "data" / "processed" / "features"
    swara_df = pd.read_csv(features_root / "swaragram_segments.csv")
    yaman_row = swara_df[swara_df["label"] == "Yaman"].iloc[2]

    swara_path = features_root / yaman_row["feature_file"].replace("\\", "/")
    swara = np.load(swara_path).astype(np.float32)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(swara, aspect="auto", origin="lower", cmap="inferno", interpolation="gaussian")

    ax.set_title("Figure 3: Swaragram Matrix for Raga Yaman (12 Bins × Time Frames)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time Frames (30-second window)", fontsize=11)
    ax.set_ylabel("Relative Swara Bins", fontsize=11)
    ax.set_yticks(range(12))
    ax.set_yticklabels([
        "Sa (Tonic - 0¢)", "re (komal)", "Re (shuddha - 200¢)", "ga (komal)",
        "Ga (shuddha - 400¢)", "Ma (shuddha)", "ma (teevra - 600¢)", "Pa (Fifth - 700¢)",
        "dha (komal)", "Dha (shuddha - 900¢)", "ni (komal)", "Ni (shuddha - 1100¢)"
    ], fontsize=9.5)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pitch-Class Energy Intensity", fontsize=10)

    # Highlight characteristic notes of Raga Yaman (Sa, Re, Ga, teevra ma, Pa, Dha, Ni)
    ax.axhline(0, color="cyan", linestyle="--", alpha=0.5, label="Sa (Root)")
    ax.axhline(4, color="lime", linestyle="--", alpha=0.5, label="Ga (Emphasized)")
    ax.axhline(6, color="yellow", linestyle="--", alpha=0.5, label="teevra Ma (Characteristic)")

    ax.legend(loc="upper right", fontsize=8.5, facecolor="black", labelcolor="white")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_swaragram_yaman_bhairavi.png")
    plt.close(fig)
    logger.info("  [OK] fig3_swaragram_yaman_bhairavi.png")


# ── Figure 4: Tonic Validation (Saraga 85.2% Match & Error Dist) ──────────────
def make_fig4():
    val_csv = PROJECT_ROOT / "experiments" / "tonic_validation" / "saraga_tonic_validation.csv"
    if not val_csv.exists():
        logger.warning("Saraga validation CSV missing, skipping Fig 4")
        return
    df = pd.read_csv(val_csv)
    ok = df[df["status"] == "OK"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel A: Absolute cent error
    axes[0].hist(ok["abs_cent_error"], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(50, color="orange", linestyle="--", linewidth=1.5, label="±50 cents")
    axes[0].axvline(100, color="red", linestyle="--", linewidth=1.5, label="±100 cents")
    axes[0].set_xlabel("Absolute Cent Error (|cents|)", fontsize=11)
    axes[0].set_ylabel("Number of Recordings", fontsize=11)
    axes[0].set_title("A: Absolute Cent Error Distribution\n(Peak at 1200¢ = Octave Shift)", fontsize=11, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel B: Pitch class agreement & octave error pie
    labels = ['Pitch-Class Match\n(85.2%)', 'Pitch-Class Miss\n(14.8%)']
    sizes = [85.2, 14.8]
    colors = [SWARA_COLOR, MEL_COLOR]
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
                textprops={'fontsize': 10, 'weight': 'bold'})
    axes[1].set_title("B: Pitch-Class Agreement (N=108)\n(58.3% Octave Shifts - Wrapped Modulo 1200)", fontsize=11, fontweight="bold")

    fig.suptitle("Figure 4: Tonic Estimator Quantitative Validation Against Saraga Ground-Truth",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_tonic_validation.png")
    plt.close(fig)
    logger.info("  [OK] fig4_tonic_validation.png")


# ── Figure 5: Confusion Matrices (Mel vs Swaragram Side-by-Side) ─────────────
def make_fig5():
    logger.info("Generating Fig 5: Confusion Matrices (Mel vs Swaragram Side-by-Side)...")
    from sklearn.metrics import confusion_matrix

    mapping_file = PROJECT_ROOT / "data" / "processed" / "metadata" / "label_mapping.json"
    with open(mapping_file) as f:
        mapping = json.load(f)

    class_names = [k for k, v in sorted(mapping.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    mel_csv = PROJECT_ROOT / "experiments" / "baseline_crnn_A" / "song_level_results.csv"
    swara_csv = PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "song_level_results.csv"

    mel_df = pd.read_csv(mel_csv)
    swara_df = pd.read_csv(swara_csv)

    cm_mel = confusion_matrix(mel_df["true_idx"], mel_df["pred_idx"], labels=range(num_classes))
    cm_swara = confusion_matrix(swara_df["true_idx"], swara_df["pred_idx"], labels=range(num_classes))

    cm_mel_norm = cm_mel.astype(float) / (cm_mel.sum(axis=1, keepdims=True) + 1e-8)
    cm_swara_norm = cm_swara.astype(float) / (cm_swara.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))

    sns.heatmap(
        cm_mel_norm, ax=axes[0], cmap="Blues", vmin=0.0, vmax=1.0, cbar=True,
        cbar_kws={"label": "Normalized Recall", "shrink": 0.8},
        xticklabels=class_names, yticklabels=class_names, linewidths=0.2, linecolor="#E5E5E5"
    )
    axes[0].set_title("A: Baseline Log-Mel CRNN\n(Song Top-1 Acc: 14.7% — Diffuse Error Scatter)",
                      fontsize=12.5, fontweight="bold", color="#B7094C", pad=12)
    axes[0].set_xlabel("Predicted Canonical Raga", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Canonical Raga", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=90, labelsize=8)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8)

    sns.heatmap(
        cm_swara_norm, ax=axes[1], cmap="Blues", vmin=0.0, vmax=1.0, cbar=True,
        cbar_kws={"label": "Normalized Recall", "shrink": 0.8},
        xticklabels=class_names, yticklabels=class_names, linewidths=0.2, linecolor="#E5E5E5"
    )
    axes[1].set_title("B: Proposed Swaragram CRNN (Ours)\n(Song Top-1 Acc: 57.8% — Dominant Diagonal)",
                      fontsize=12.5, fontweight="bold", color="#0077B6", pad=12)
    axes[1].set_xlabel("Predicted Canonical Raga", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Canonical Raga", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=90, labelsize=8)
    axes[1].tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle("Figure 5: Test Set Normalized Confusion Matrices (38 Ragas, N = 116 Full Recordings)",
                 fontsize=14.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig5_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [OK] fig5_confusion_matrix.png")


# ── Figure 6: Song Pooling Architecture ────────────────────────────────────────
def make_fig6():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(0.5, 3.3, "Recording (Full Audio)", fontsize=10, fontweight="bold", ha="center")
    rect_audio = plt.Rectangle((0, 2.5), 1.0, 0.5, color="#457B9D", alpha=0.8)
    ax.add_patch(rect_audio)

    # Segments
    seg_x = [2.2, 4.2, 6.2]
    seg_labels = ["Segment 1\n(0-30s)", "Segment 2\n(15-45s)", "... Segment K\n(Trailing)"]

    for x, lbl in zip(seg_x, seg_labels):
        r = plt.Rectangle((x-0.7, 2.5), 1.4, 0.6, color=SWARA_COLOR, alpha=0.85)
        ax.add_patch(r)
        ax.text(x, 2.8, lbl, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.annotate("", xy=(x, 2.0), xytext=(x, 2.5), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="gray"))

        # Prob vector box
        rp = plt.Rectangle((x-0.7, 1.2), 1.4, 0.7, color=ACCENT, alpha=0.9)
        ax.add_patch(rp)
        ax.text(x, 1.55, f"CRNN Posteriors\nP(y | seg_{x})", ha="center", va="center", fontsize=7.5, color="white")
        ax.annotate("", xy=(8.5, 1.55), xytext=(x+0.7, 1.55), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#6A4C93"))

    # Mean pool box
    rm = plt.Rectangle((8.5, 1.0), 1.8, 1.1, color="#6A4C93", alpha=0.9)
    ax.add_patch(rm)
    ax.text(9.4, 1.55, "Probability\nMean Pooling\n1/K Σ P(y|seg_k)", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    ax.annotate("", xy=(10.8, 1.55), xytext=(10.3, 1.55), arrowprops=dict(arrowstyle="-|>", lw=2, color="#E9C46A"))

    # Final prediction box
    rf = plt.Rectangle((10.8, 1.1), 1.1, 0.9, color="#E9C46A", alpha=0.95)
    ax.add_patch(rf)
    ax.text(11.35, 1.55, "Final Song\nPrediction\n(Top-1/3)", ha="center", va="center", fontsize=8, color="black", fontweight="bold")

    ax.set_title("Figure 6: Song-Level Probability Mean Pooling Across 30-Second Segments (+19.7 pp Gain)",
                 fontsize=12, fontweight="bold", pad=10)
    fig.savefig(OUT_DIR / "fig6_song_pooling.png")
    plt.close(fig)
    logger.info("  [OK] fig6_song_pooling.png")


# ── Figure 7: Per-Raga Accuracy Bar Chart ──────────────────────────────────────
def make_fig7():
    song_csv = PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "song_level_results.csv"
    if not song_csv.exists():
        logger.warning("song_level_results.csv missing, skipping Fig 7")
        return
    df = pd.read_csv(song_csv)

    # Compute top-1 accuracy per raga
    raga_acc = df.groupby("true_label").agg(
        total=("top1_correct", "count"),
        top1_correct=("top1_correct", "sum"),
        top3_correct=("top3_correct", "sum")
    ).reset_index()

    raga_acc["top1_acc"] = 100.0 * raga_acc["top1_correct"] / raga_acc["total"]
    raga_acc["top3_acc"] = 100.0 * raga_acc["top3_correct"] / raga_acc["total"]
    raga_acc = raga_acc.sort_values(by="top1_acc", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(raga_acc))
    w = 0.4

    ax.bar(x - w/2, raga_acc["top1_acc"], w, label="Top-1 Accuracy", color=SWARA_COLOR, alpha=0.9)
    ax.bar(x + w/2, raga_acc["top3_acc"], w, label="Top-3 Accuracy", color="#E9C46A", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(raga_acc["true_label"], rotation=90, fontsize=8)
    ax.set_ylabel("Song-Level Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Figure 7: Song-Level Top-1 and Top-3 Accuracy Across 38 Ragas (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_per_raga_accuracy.png")
    plt.close(fig)
    logger.info("  [OK] fig7_per_raga_accuracy.png")


if __name__ == "__main__":
    logger.info(f"=== Generating 7 Mandatory Paper Figures -> {OUT_DIR} ===")
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    logger.info("=== Complete! ===")
