"""
generate_all_figures.py

Generates ALL publication figures for the paper in one shot.
Output directory: experiments/paper_figures/

Figures produced:
  fig1_pipeline.png          – System pipeline overview
  fig2_feature_comparison.png – Mel Spectrogram vs Swaragram side-by-side
  fig3_training_curves.png   – Training curves (all experiments, 2-panel)
  fig4_results_barchart.png  – Summary bar chart (seg + song level)
  fig5_confusion_matrix.png  – Song-level confusion matrix (Swaragram)
  fig6_top3_table.png        – Qualitative Top-3 prediction table
  fig7_tonic_histogram.png   – Two tonic estimation examples side-by-side
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
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "experiments" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
})

MEL_COLOR   = "#E63946"   # red
SWARA_COLOR = "#2A9D8F"   # teal
ACCENT      = "#264653"   # dark


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: System Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fig1_pipeline():
    logger.info("Generating Fig 1: System Pipeline...")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("Raw Audio\n(Folk Music)", 0.7,  "#457B9D"),
        ("Pitch\nExtraction\n(YIN)",       2.5,  "#1D3557"),
        ("Tonic\nEstimation\n(Sa)",         4.3,  "#1D3557"),
        ("Swaragram\nGeneration\n(12-bin)", 6.1,  SWARA_COLOR),
        ("CRNN +\nBiLSTM\nEncoder",        8.2,  ACCENT),
        ("Song-Level\nAggregation\n(Mean Pool)", 10.3, "#6A4C93"),
        ("Raag\nPrediction\n(Top-3)",      12.2, "#E9C46A"),
    ]

    box_w, box_h = 1.5, 1.9
    y_center = 2.0
    for label, x, color in stages:
        rect = plt.Rectangle((x - box_w / 2, y_center - box_h / 2),
                              box_w, box_h, color=color, alpha=0.9,
                              zorder=3, linewidth=1.5, edgecolor="white")
        ax.add_patch(rect)
        ax.text(x, y_center, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold", zorder=4,
                multialignment="center")

    # Arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][1] + box_w / 2
        x2 = stages[i + 1][1] - box_w / 2
        ax.annotate("", xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle="-|>", color="gray",
                                   lw=1.8), zorder=2)

    ax.set_title("System Pipeline — Swaragram-Based Melodic Origin Estimation",
                 fontsize=13, fontweight="bold", pad=12)
    fig.savefig(OUT_DIR / "fig1_pipeline.png")
    plt.close(fig)
    logger.info("  [OK] fig1_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Feature Comparison (Mel vs Swaragram)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_feature_comparison():
    logger.info("Generating Fig 2: Feature Comparison...")
    features_root = PROJECT_ROOT / "data" / "processed" / "features"

    # Find a paired mel/swaragram segment for the same recording
    mel_df   = pd.read_csv(features_root / "mel_segments.csv")
    swara_df = pd.read_csv(features_root / "swaragram_segments.csv")

    # Use trf_0028 (Asawari, test set, 21 segments) — guaranteed interesting
    mel_row   = mel_df[mel_df["recording_id"] == "trf_0028"].iloc[5]
    swara_row = swara_df[swara_df["recording_id"] == "trf_0028"].iloc[5]
    raag_name = mel_row["label"]

    mel_path   = features_root / mel_row["feature_file"].replace("\\", "/")
    swara_path = features_root / swara_row["feature_file"].replace("\\", "/")

    mel   = np.load(mel_path).astype(np.float32)    # (128, T)
    swara = np.load(swara_path).astype(np.float32)  # (12, T)

    # Normalize Mel for display
    with open(PROJECT_ROOT / "data" / "processed" / "metadata" / "dataset_stats.json") as f:
        stats = json.load(f)
    mel_disp = (mel - stats["train_mean"]) / (stats["train_std"] + 1e-8)

    SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # — Mel spectrogram —
    im0 = axes[0].imshow(mel_disp, aspect="auto", origin="lower",
                         cmap="magma", interpolation="nearest")
    axes[0].set_title(f"Mel Spectrogram\n(Raag: {raag_name})",
                      fontsize=12, fontweight="bold", color=MEL_COLOR)
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Mel Bin (0–127)")
    plt.colorbar(im0, ax=axes[0], label="Normalized Amplitude")
    axes[0].text(0.02, 0.97, "Encodes absolute pitch + timbre\n=> Overfits to singer identity",
                 transform=axes[0].transAxes, fontsize=8.5, va="top",
                 color="white",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=MEL_COLOR, alpha=0.8))

    # — Swaragram —
    im1 = axes[1].imshow(swara, aspect="auto", origin="lower",
                         cmap="YlOrRd", interpolation="nearest",
                         vmin=0, vmax=swara.max())
    axes[1].set_title(f"Swaragram (Pitch-Class Histogram)\n(Raag: {raag_name})",
                      fontsize=12, fontweight="bold", color=SWARA_COLOR)
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Pitch Class (Swara)")
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels(SWARA_NAMES, fontsize=9)
    plt.colorbar(im1, ax=axes[1], label="Relative Energy")
    axes[1].text(0.02, 0.97, "Encodes relative pitch classes\n=> Invariant to singer/tonic",
                 transform=axes[1].transAxes, fontsize=8.5, va="top",
                 color="white",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=SWARA_COLOR, alpha=0.8))

    fig.suptitle("Feature Representations for the Same Audio Segment",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_feature_comparison.png")
    plt.close(fig)
    logger.info("  [OK] fig2_feature_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Training Curves — Both Experiments
# ─────────────────────────────────────────────────────────────────────────────
def fig3_training_curves():
    logger.info("Generating Fig 3: Training Curves...")

    mel_hist   = pd.read_csv(PROJECT_ROOT / "experiments" / "baseline_crnn_A" / "history.csv")
    swara_hist = pd.read_csv(PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "history.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # — Loss —
    ax = axes[0]
    ax.plot(mel_hist["Epoch"],   mel_hist["Train Loss"],   color=MEL_COLOR,   lw=2,
            linestyle="--", label="Mel — Train")
    ax.plot(mel_hist["Epoch"],   mel_hist["Val Loss"],     color=MEL_COLOR,   lw=2,
            label="Mel — Val")
    ax.plot(swara_hist["Epoch"], swara_hist["Train Loss"], color=SWARA_COLOR, lw=2,
            linestyle="--", label="Swaragram — Train")
    ax.plot(swara_hist["Epoch"], swara_hist["Val Loss"],   color=SWARA_COLOR, lw=2,
            label="Swaragram — Val")
    ax.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # — Validation Accuracy —
    ax = axes[1]
    ax.plot(mel_hist["Epoch"],   mel_hist["Train Acc"],   color=MEL_COLOR,   lw=2,
            linestyle="--", label="Mel — Train Acc")
    ax.plot(mel_hist["Epoch"],   mel_hist["Val Acc"],     color=MEL_COLOR,   lw=2.5,
            label="Mel — Val Acc")
    ax.plot(swara_hist["Epoch"], swara_hist["Train Acc"], color=SWARA_COLOR, lw=2,
            linestyle="--", label="Swaragram — Train Acc")
    ax.plot(swara_hist["Epoch"], swara_hist["Val Acc"],   color=SWARA_COLOR, lw=2.5,
            label="Swaragram — Val Acc")
    if "Val Top3" in swara_hist.columns:
        ax.plot(swara_hist["Epoch"], swara_hist["Val Top3"], color=SWARA_COLOR, lw=1.5,
                linestyle=":", label="Swaragram — Val Top-3")
    ax.axhline(1/38, color="gray", linestyle=":", lw=1.2, label="Random chance (1/38)")
    ax.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Learning Curves — Mel Spectrogram vs Swaragram CRNN",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_training_curves.png")
    plt.close(fig)
    logger.info("  [OK] fig3_training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Results Summary Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def fig4_results_barchart():
    logger.info("Generating Fig 4: Results Bar Chart...")

    data = {
        "Experiment": [
            "Random\nChance",
            "Mel CRNN\n(Segment)",
            "Mel CRNN\n(Song-Level)",
            "Swaragram CRNN\n(Segment)",
            "Swaragram CRNN\n(Song-Level)",
        ],
        "Top-1 Acc": [7.9,  8.0,  14.7, 38.1, 57.8],
        "Top-3 Acc": [23.7, 20.0, 24.1, 56.9, 75.9],
        "Macro F1":  [0.0,  8.0,  12.0, 35.7, 55.2],
        "color": ["#ADB5BD", MEL_COLOR, MEL_COLOR, SWARA_COLOR, SWARA_COLOR],
        "alpha": [0.6, 0.5, 0.9, 0.5, 0.9],
    }
    df = pd.DataFrame(data)

    x = np.arange(len(df))
    w = 0.26

    fig, ax = plt.subplots(figsize=(13, 6))

    # Draw bars individually so per-bar alpha works
    for i, row in df.iterrows():
        ax.bar(x[i] - w, row["Top-1 Acc"], w, color=row["color"], alpha=row["alpha"],
               edgecolor="white", linewidth=0.8)
        ax.bar(x[i],     row["Top-3 Acc"], w, color=row["color"], alpha=min(row["alpha"]+0.1,1),
               edgecolor="white", linewidth=0.8, hatch="//")
        ax.bar(x[i] + w, row["Macro F1"],  w, color=row["color"], alpha=row["alpha"],
               edgecolor="white", linewidth=0.8, hatch="xx")

    # Value labels
    for bar in ax.patches:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

    ax.axhline(7.9, color="gray", linestyle=":", lw=1.5, label="Random chance (7.9%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Experiment"], fontsize=10)
    ax.set_ylabel("Accuracy / F1 Score (%)")
    ax.set_ylim(0, 88)
    ax.set_title("Performance Comparison Across Experiments",
                 fontsize=13, fontweight="bold")

    mel_patch   = mpatches.Patch(color=MEL_COLOR,   label="Mel Spectrogram")
    swara_patch = mpatches.Patch(color=SWARA_COLOR, label="Swaragram (ours)")
    ax.legend(handles=[mel_patch, swara_patch,
                       mpatches.Patch(color="gray", label="Random Chance")],
              loc="upper left", fontsize=10)

    # Metric legend (pattern)
    top1  = mpatches.Patch(facecolor="silver", edgecolor="black", label="Top-1 Acc (solid)")
    top3  = mpatches.Patch(facecolor="silver", edgecolor="black", hatch="//", label="Top-3 Acc (//)")
    macro = mpatches.Patch(facecolor="silver", edgecolor="black", hatch="xx", label="Macro F1 (xx)")
    ax.legend(handles=[mel_patch, swara_patch, top1, top3, macro],
              loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_results_barchart.png")
    plt.close(fig)
    logger.info("  [OK] fig4_results_barchart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Song-Level Confusion Matrix (Swaragram)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_confusion_matrix():
    logger.info("Generating Fig 5: Song-Level Confusion Matrix...")
    import shutil
    src = PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "song_level_confusion_matrix.png"
    dst = OUT_DIR / "fig5_confusion_matrix.png"
    shutil.copy2(src, dst)
    logger.info("  [OK] fig5_confusion_matrix.png  (copied from swaragram_crnn_A)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Top-3 Qualitative Prediction Table
# ─────────────────────────────────────────────────────────────────────────────
def fig6_top3_table():
    logger.info("Generating Fig 6: Top-3 Qualitative Table...")
    df = pd.read_csv(PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "top3_qualitative_table.csv")

    # Rename columns for display
    df = df.rename(columns={
        "True Raag"    : "True Raag",
        "Pred 1 (prob)": "1st Prediction",
        "Pred 2 (prob)": "2nd Prediction",
        "Pred 3 (prob)": "3rd Prediction",
        "Top-1"        : "Top-1",
        "Top-3"        : "Top-3",
    })

    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.axis("off")

    col_widths = [0.18, 0.24, 0.24, 0.24, 0.05, 0.05]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(df.columns)):
        table[0, j].set_facecolor(ACCENT)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by correctness
    for i in range(len(df)):
        top1_ok = df.iloc[i]["Top-1"] == "[OK]"
        top3_ok = df.iloc[i]["Top-3"] == "[OK]"
        bg = "#D4EDDA" if top1_ok else ("#FFF3CD" if top3_ok else "#F8D7DA")
        for j in range(len(df.columns)):
            table[i + 1, j].set_facecolor(bg)

    ax.set_title("Qualitative Top-3 Predictions — Swaragram CRNN (Test Set Segments)",
                 fontsize=12, fontweight="bold", pad=20)

    legend_patches = [
        mpatches.Patch(color="#D4EDDA", label="Top-1 Correct"),
        mpatches.Patch(color="#FFF3CD", label="Top-3 Correct"),
        mpatches.Patch(color="#F8D7DA", label="Not in Top-3"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              bbox_to_anchor=(1.0, -0.05), fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_top3_table.png")
    plt.close(fig)
    logger.info("  [OK] fig6_top3_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Tonic Estimation Examples
# ─────────────────────────────────────────────────────────────────────────────
def fig7_tonic_histogram():
    logger.info("Generating Fig 7: Tonic Estimation Histograms...")
    import librosa
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.swaragram.pitch_extractor import PitchExtractor
    from src.swaragram.tonic_estimator import TonicEstimator
    from src.preprocessing.audio_utils import load_and_resample_audio, resolve_audio_path

    SR = 22050
    extractor = PitchExtractor(sr=SR, hop_length=512)
    tonic_est = TonicEstimator()

    # Two representative recordings from tonic_eval results
    # Use metadata CSV to find audio paths
    splits_dir = PROJECT_ROOT / "data" / "processed" / "metadata"
    all_csvs = list(splits_dir.rglob("*.csv"))
    meta = pd.concat([pd.read_csv(p) for p in all_csvs], ignore_index=True)
    meta = meta.dropna(subset=["audio_path"]).drop_duplicates("recording_id")

    examples = [
        ("trf_0082",  "Hindustani Male Voice\n(Sa: D2, 73.8 Hz)"),
        ("imrd_0047", "Carnatic Voice\n(Sa: C#4, 272.8 Hz)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    SWARA_COLOR_LIGHT = "#52B788"

    for ax, (rec_id, subtitle) in zip(axes, examples):
        row = meta[meta["recording_id"] == rec_id]
        if row.empty:
            ax.set_title(f"{rec_id} — not found")
            continue

        audio_path = resolve_audio_path(row.iloc[0]["audio_path"])
        audio, _ = load_and_resample_audio(audio_path, target_sr=SR, normalize=True)
        audio = audio[:SR * 60]

        f0 = extractor.extract(audio)
        tonic_hz = tonic_est.estimate(f0)
        tonic_note = librosa.midi_to_note(round(float(librosa.hz_to_midi(tonic_hz))))

        f0_valid = f0[(~np.isnan(f0)) & (f0 > 0)]
        midi_vals = librosa.hz_to_midi(f0_valid)
        tonic_midi = float(librosa.hz_to_midi(tonic_hz))

        ax.hist(midi_vals, bins=72, color=SWARA_COLOR_LIGHT, edgecolor="none", alpha=0.85)
        ax.axvline(tonic_midi, color="#E63946", lw=2.5,
                   label=f"Estimated Sa = {tonic_hz:.1f} Hz ({tonic_note})")
        ax.set_title(f"Recording: {rec_id}\n{subtitle}", fontsize=11, fontweight="bold")
        ax.set_xlabel("MIDI Note")
        ax.set_ylabel("Frame Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Tonic (Sa) Estimation via Pitch Histogram Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_tonic_histogram.png")
    plt.close(fig)
    logger.info("  [OK] fig7_tonic_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"=== Generating all paper figures -> {OUT_DIR} ===\n")

    fig1_pipeline()
    fig2_feature_comparison()
    fig3_training_curves()
    fig4_results_barchart()
    fig5_confusion_matrix()
    fig6_top3_table()
    fig7_tonic_histogram()

    logger.info(f"\n=== All figures complete ===")
    logger.info(f"Output: {OUT_DIR}")
    figs = sorted(OUT_DIR.glob("*.png"))
    for f in figs:
        logger.info(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
