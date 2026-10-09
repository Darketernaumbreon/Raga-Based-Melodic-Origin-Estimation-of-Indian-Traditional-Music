"""
generate_paper_package.py

Assembles the complete paper submission package:

  paper/
    figures/   - All publication figures (Fig 1-10)
    tables/    - CSV metric tables + classification report
    appendix/  - Training logs, evaluation logs

New figures generated here (complementing existing ones):
  fig_dataset_distribution.png  - Recordings per raag (horizontal bar)
  fig_progression.png           - Accuracy progression line chart
  fig_pooling_diagram.png       - Song-level aggregation conceptual diagram
  fig_feature_table.png         - Mel vs Swaragram properties comparison
  fig_segment_vs_song.png       - Segment vs song-level grouped bar chart
"""
import sys
import shutil
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAPER_DIR   = PROJECT_ROOT / "paper"
FIG_DIR     = PAPER_DIR / "figures"
TABLE_DIR   = PAPER_DIR / "tables"
APP_DIR     = PAPER_DIR / "appendix"
PREV_FIGS   = PROJECT_ROOT / "experiments" / "paper_figures"

for d in [FIG_DIR, TABLE_DIR, APP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "figure.dpi"       : 150,
})

MEL_COLOR   = "#E63946"
SWARA_COLOR = "#2A9D8F"
ACCENT      = "#264653"
GREY        = "#6C757D"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Copy existing figures
# ─────────────────────────────────────────────────────────────────────────────
def copy_existing_figures():
    logger.info("Copying existing figures...")
    mapping = {
        "fig1_pipeline.png"           : "fig1_pipeline.png",
        "fig2_feature_comparison.png" : "fig3_mel_vs_swaragram.png",
        "fig3_training_curves.png"    : "fig4_training_curves.png",
        "fig4_results_barchart.png"   : "fig6_results_comparison.png",
        "fig5_confusion_matrix.png"   : "fig7_confusion_matrix.png",
        "fig6_top3_table.png"         : "fig8_qualitative_predictions.png",
        "fig7_tonic_histogram.png"    : "fig_tonic_estimation.png",
    }
    for src_name, dst_name in mapping.items():
        src = PREV_FIGS / src_name
        if src.exists():
            shutil.copy2(src, FIG_DIR / dst_name)
            logger.info(f"  [OK] {dst_name}")
        else:
            logger.warning(f"  [!] Not found: {src_name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW Fig 2: Dataset Distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_dataset_distribution():
    logger.info("Generating Fig 2: Dataset Distribution...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "swaragram_segments.csv")
    counts = df.groupby("label")["recording_id"].nunique().sort_values(ascending=True)

    # Color by thaat family (approximate grouping)
    kalyan  = {"Yaman", "Yaman Kalyan", "Bhoopali", "Bhairav Bahar", "Kedar", "Shree", "Puriya", "Marwa", "Multani"}
    bilaval = {"Bilaval", "Bihag", "Desh", "Khamaj", "Kafi", "Bhimpalasi", "Bageshree"}
    bhairav = {"Bhairav", "Bibhas", "Gauri", "Nat Bhairav", "Ahir Bhairav", "Bilaskhani Todi"}
    asawari = {"Asawari", "Bhairavi", "Malkauns", "Adana", "Abhogi"}
    poorvi  = {"Poorvi", "Lalit", "Bhatiyar"}

    def get_color(name):
        if name in kalyan:  return "#4361EE"
        if name in bilaval: return "#3A86FF"
        if name in bhairav: return "#8338EC"
        if name in asawari: return "#E63946"
        if name in poorvi:  return "#FB8500"
        return SWARA_COLOR

    colors = [get_color(r) for r in counts.index]

    fig, ax = plt.subplots(figsize=(10, 12))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.75)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    ax.set_xlabel("Number of Recordings", fontsize=11)
    ax.set_title("Dataset Distribution — Recordings per Raag\n"
                 f"(38 Raags, {counts.sum()} Total Recordings)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, counts.max() + 5)
    ax.axvline(counts.mean(), color="gray", linestyle="--", lw=1.2,
               label=f"Mean: {counts.mean():.1f}")
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend for thaat families
    legend_items = [
        mpatches.Patch(color="#4361EE", label="Kalyan Thaat"),
        mpatches.Patch(color="#3A86FF", label="Bilaval / Kafi Thaat"),
        mpatches.Patch(color="#8338EC", label="Bhairav Thaat"),
        mpatches.Patch(color="#E63946", label="Asawari / Bhairavi Thaat"),
        mpatches.Patch(color="#FB8500", label="Poorvi Thaat"),
        mpatches.Patch(color=SWARA_COLOR, label="Other"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_dataset_distribution.png")
    plt.close(fig)
    logger.info("  [OK] fig2_dataset_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# NEW Fig 5: Segment vs Song-Level Grouped Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_segment_vs_song():
    logger.info("Generating Fig 5: Segment vs Song-Level Comparison...")

    labels    = ["Mel CRNN", "Swaragram CRNN"]
    seg_top1  = [8.0,  38.1]
    song_top1 = [14.7, 57.8]
    seg_top3  = [20.0, 56.9]
    song_top3 = [24.1, 75.9]

    x = np.array([0, 1])
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    # Top-1
    ax = axes[0]
    ax.bar(x - w/2, seg_top1,  w*1.8, label="Segment-Level", color=[MEL_COLOR, SWARA_COLOR],
           alpha=0.5, edgecolor="white")
    ax.bar(x + w/2, song_top1, w*1.8, label="Song-Level",    color=[MEL_COLOR, SWARA_COLOR],
           alpha=0.95, edgecolor="white", hatch="//")
    ax.axhline(7.9, color=GREY, linestyle=":", lw=1.5, label="Random (7.9%)")
    for i, (sv, sl) in enumerate(zip(seg_top1, song_top1)):
        ax.text(x[i] - w/2, sv + 0.8, f"{sv}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(x[i] + w/2, sl + 0.8, f"{sl}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Top-1 Accuracy\nSegment vs Song-Level", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 70)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # Top-3
    ax = axes[1]
    ax.bar(x - w/2, seg_top3,  w*1.8, label="Segment-Level", color=[MEL_COLOR, SWARA_COLOR],
           alpha=0.5, edgecolor="white")
    ax.bar(x + w/2, song_top3, w*1.8, label="Song-Level",    color=[MEL_COLOR, SWARA_COLOR],
           alpha=0.95, edgecolor="white", hatch="//")
    ax.axhline(23.7, color=GREY, linestyle=":", lw=1.5, label="Random Top-3 (23.7%)")
    for i, (sv, sl) in enumerate(zip(seg_top3, song_top3)):
        ax.text(x[i] - w/2, sv + 0.8, f"{sv}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(x[i] + w/2, sl + 0.8, f"{sl}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Top-3 Accuracy (%)")
    ax.set_title("Top-3 Accuracy\nSegment vs Song-Level", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 90)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Effect of Song-Level Probability Pooling on Classification Accuracy",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_segment_vs_song.png")
    plt.close(fig)
    logger.info("  [OK] fig5_segment_vs_song.png")


# ─────────────────────────────────────────────────────────────────────────────
# NEW Fig 9: Accuracy Progression (line chart)
# ─────────────────────────────────────────────────────────────────────────────
def fig_progression():
    logger.info("Generating Fig 9: Accuracy Progression...")

    steps  = ["Random\nChance", "Mel CRNN\n(Segment)", "Mel CRNN\n(Song-Level)",
              "Swaragram\n(Segment)", "Swaragram\n(Song-Level)"]
    top1   = [7.9,  8.0,  14.7, 38.1, 57.8]
    top3   = [23.7, 20.0, 24.1, 56.9, 75.9]
    colors = [GREY, MEL_COLOR, MEL_COLOR, SWARA_COLOR, SWARA_COLOR]
    sizes  = [60, 80, 100, 100, 140]

    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(x, top1, color=ACCENT, lw=2.5, zorder=2, label="Top-1 Accuracy")
    ax.plot(x, top3, color="#6A4C93", lw=2.5, zorder=2, linestyle="--", label="Top-3 Accuracy")

    for i, (y1, y3, c, s) in enumerate(zip(top1, top3, colors, sizes)):
        ax.scatter(x[i], y1, color=c, s=s, zorder=5, edgecolors="white", linewidth=1.5)
        ax.scatter(x[i], y3, color=c, s=s, zorder=5, edgecolors="white", linewidth=1.5, marker="D")
        ax.text(x[i], y1 + 2.5, f"{y1}%", ha="center", fontsize=9,
                fontweight="bold", color=c)
        ax.text(x[i], y3 + 2.5, f"{y3}%", ha="center", fontsize=9,
                fontweight="bold", color=c)

    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 88)
    ax.set_title("Research Progression — Accuracy Improvement Across Experiments",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Shaded regions
    ax.axvspan(-0.5, 0.5,  alpha=0.06, color=GREY,        label="Baseline")
    ax.axvspan(0.5,  2.5,  alpha=0.06, color=MEL_COLOR,   label="Mel Experiments")
    ax.axvspan(2.5,  4.5,  alpha=0.06, color=SWARA_COLOR, label="Swaragram Experiments")

    ax.legend(fontsize=9, loc="upper left")

    # Annotation arrows for key gains
    ax.annotate("", xy=(4, 57.8), xytext=(2, 14.7),
                arrowprops=dict(arrowstyle="-|>", color=SWARA_COLOR, lw=1.5,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(3.1, 32, "+43.1%\nSwaragram\nvs Mel Song", fontsize=8.5,
            color=SWARA_COLOR, fontweight="bold", ha="center")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_progression.png")
    plt.close(fig)
    logger.info("  [OK] fig9_progression.png")


# ─────────────────────────────────────────────────────────────────────────────
# NEW Fig 10: Song-Level Aggregation Concept Diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_pooling_diagram():
    logger.info("Generating Fig 10: Song-Level Pooling Diagram...")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Example: Ramkali recording
    segments = [
        ("Seg 1", "Ramkali", 0.71, True),
        ("Seg 2", "Ramkali", 0.64, True),
        ("Seg 3", "Durga",   0.35, False),
        ("Seg 4", "Ramkali", 0.83, True),
        ("Seg 5", "Ramkali", 0.77, True),
    ]

    seg_ys = [5.0, 4.1, 3.2, 2.3, 1.4]
    for (seg_name, pred, conf, correct), y in zip(segments, seg_ys):
        c = SWARA_COLOR if correct else MEL_COLOR
        # Segment box
        rect = plt.Rectangle((0.3, y - 0.3), 1.5, 0.6, color="#DEF2F1",
                              edgecolor=c, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.05, y, seg_name, ha="center", va="center", fontsize=9, fontweight="bold")

        # Arrow
        ax.annotate("", xy=(2.8, y), xytext=(1.8, y),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2))

        # CRNN box
        rect2 = plt.Rectangle((2.8, y - 0.3), 1.6, 0.6, color="#1D3557",
                               edgecolor="white", linewidth=1)
        ax.add_patch(rect2)
        ax.text(3.6, y, "CRNN", ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")

        # Arrow to prediction
        ax.annotate("", xy=(5.4, y), xytext=(4.4, y),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2))

        # Prediction box
        pred_color = SWARA_COLOR if correct else MEL_COLOR
        rect3 = plt.Rectangle((5.4, y - 0.3), 3.5, 0.6, color=pred_color,
                               alpha=0.2, edgecolor=pred_color, linewidth=1.5)
        ax.add_patch(rect3)
        mark = "[OK]" if correct else "[X]"
        ax.text(7.15, y, f"{pred} ({conf:.2f})  {mark}",
                ha="center", va="center", fontsize=9,
                color=ACCENT, fontweight="bold")

    # Pooling arrow — all predictions converge
    ax.annotate("", xy=(9.6, 3.2), xytext=(9.1, 5.0),
                arrowprops=dict(arrowstyle="-|>", color="#6A4C93", lw=2))
    ax.annotate("", xy=(9.6, 3.2), xytext=(9.1, 4.1),
                arrowprops=dict(arrowstyle="-|>", color="#6A4C93", lw=2))
    ax.annotate("", xy=(9.6, 3.2), xytext=(9.1, 3.2),
                arrowprops=dict(arrowstyle="-|>", color="#6A4C93", lw=2))
    ax.annotate("", xy=(9.6, 3.2), xytext=(9.1, 2.3),
                arrowprops=dict(arrowstyle="-|>", color="#6A4C93", lw=2))
    ax.annotate("", xy=(9.6, 3.2), xytext=(9.1, 1.4),
                arrowprops=dict(arrowstyle="-|>", color="#6A4C93", lw=2))

    # Pooling box
    rect_pool = plt.Rectangle((9.6, 2.7), 1.8, 1.0, color="#6A4C93",
                               edgecolor="white", linewidth=1.5)
    ax.add_patch(rect_pool)
    ax.text(10.5, 3.2, "Mean\nPool", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

    # Final output arrow
    ax.annotate("", xy=(12.2, 3.2), xytext=(11.4, 3.2),
                arrowprops=dict(arrowstyle="-|>", color=SWARA_COLOR, lw=2))

    # Final prediction
    rect_final = plt.Rectangle((12.2, 2.7), 0.8, 1.0, color=SWARA_COLOR,
                                edgecolor="white", linewidth=1.5)
    ax.add_patch(rect_final)
    ax.text(12.6, 3.2, "Ramkali\n[OK]", ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold")

    ax.text(6.5, 5.8, "True Raag: Ramkali  |  5 segments  |  4/5 correct at segment level",
            ha="center", va="center", fontsize=10, fontweight="bold", color=ACCENT)
    ax.text(10.5, 2.1, "Errors cancel\nout via pooling", ha="center", fontsize=8.5,
            color="#6A4C93", style="italic")

    ax.set_title("Song-Level Probability Pooling — How Segment Errors Are Resolved",
                 fontsize=13, fontweight="bold", pad=15)
    fig.savefig(FIG_DIR / "fig10_pooling_diagram.png")
    plt.close(fig)
    logger.info("  [OK] fig10_pooling_diagram.png")


# ─────────────────────────────────────────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────────────────────────────────────────
def generate_tables():
    logger.info("Generating tables...")

    # Table 1: Main results
    results = pd.DataFrame([
        {"Experiment": "Random Chance",               "Feature": "—",          "Seg Top-1": 7.9,  "Seg Top-3": 23.7, "Song Top-1": "—",  "Song Top-3": "—",  "Song F1": "—"},
        {"Experiment": "Exp 1: Mel CRNN",             "Feature": "Mel",        "Seg Top-1": 8.0,  "Seg Top-3": 20.0, "Song Top-1": 14.7, "Song Top-3": 24.1, "Song F1": 12.0},
        {"Experiment": "Exp 3: Swaragram CRNN",       "Feature": "Swaragram",  "Seg Top-1": 38.1, "Seg Top-3": 56.9, "Song Top-1": 57.8, "Song Top-3": 75.9, "Song F1": 55.2},
        {"Experiment": "Exp 3B: Refined Swaragram",   "Feature": "Swaragram+", "Seg Top-1": 38.0, "Seg Top-3": 56.9, "Song Top-1": "—",  "Song Top-3": "—",  "Song F1": "—"},
        {"Experiment": "Exp 4: Hybrid (Mel+Swara)",   "Feature": "Both",       "Seg Top-1": "~8", "Seg Top-3": "~22","Song Top-1": "—",  "Song Top-3": "—",  "Song F1": "—"},
    ])
    results.to_csv(TABLE_DIR / "experiment_results.csv", index=False)
    logger.info("  [OK] experiment_results.csv")

    # Table 2: Per-raag song-level results
    song_df = pd.read_csv(PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "song_level_results.csv")
    per_raag = song_df.groupby("true_label").agg(
        recordings=("recording_id", "count"),
        top1_correct=("top1_correct", "sum"),
        top3_correct=("top3_correct", "sum"),
    ).reset_index()
    per_raag["Top-1 Acc (%)"] = (per_raag["top1_correct"] / per_raag["recordings"] * 100).round(1)
    per_raag["Top-3 Acc (%)"] = (per_raag["top3_correct"] / per_raag["recordings"] * 100).round(1)
    per_raag = per_raag.rename(columns={"true_label": "Raag", "recordings": "Test Recordings"})
    per_raag = per_raag[["Raag", "Test Recordings", "Top-1 Acc (%)", "Top-3 Acc (%)"]].sort_values("Top-1 Acc (%)", ascending=False)
    per_raag.to_csv(TABLE_DIR / "per_raag_results.csv", index=False)
    logger.info("  [OK] per_raag_results.csv")

    # Table 3: Dataset distribution
    seg_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "swaragram_segments.csv")
    dist = seg_df.groupby("label")["recording_id"].nunique().reset_index()
    dist.columns = ["Raag", "Total Recordings"]
    dist = dist.sort_values("Total Recordings", ascending=False)
    dist.to_csv(TABLE_DIR / "dataset_distribution.csv", index=False)
    logger.info("  [OK] dataset_distribution.csv")

    # Copy classification report
    report_src = PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "song_level_classification_report.txt"
    if report_src.exists():
        shutil.copy2(report_src, TABLE_DIR / "classification_report.txt")
        logger.info("  [OK] classification_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# APPENDIX
# ─────────────────────────────────────────────────────────────────────────────
def assemble_appendix():
    logger.info("Assembling appendix...")

    # Copy training logs
    for exp_name, out_name in [
        ("baseline_crnn_A",   "training_log_mel_crnn.txt"),
        ("swaragram_crnn_A",  "training_log_swaragram_crnn.txt"),
    ]:
        log_src = PROJECT_ROOT / "experiments" / exp_name / "training.log"
        if log_src.exists():
            shutil.copy2(log_src, APP_DIR / out_name)
            logger.info(f"  [OK] {out_name}")

    # Copy history CSVs
    for exp_name, out_name in [
        ("baseline_crnn_A",   "history_mel_crnn.csv"),
        ("swaragram_crnn_A",  "history_swaragram_crnn.csv"),
    ]:
        csv_src = PROJECT_ROOT / "experiments" / exp_name / "history.csv"
        if csv_src.exists():
            shutil.copy2(csv_src, APP_DIR / out_name)
            logger.info(f"  [OK] {out_name}")

    # Copy tonic evaluation results
    tonic_src = PROJECT_ROOT / "experiments" / "tonic_eval" / "tonic_evaluation_results.csv"
    if tonic_src.exists():
        shutil.copy2(tonic_src, APP_DIR / "tonic_evaluation_results.csv")
        logger.info("  [OK] tonic_evaluation_results.csv")

    # Write evaluation summary
    summary = """=== EVALUATION SUMMARY ===

Model: Swaragram CRNN (swaragram_crnn_A)
Best Checkpoint: Epoch 26
Dataset: Indian Folk Music Dataset A

SEGMENT-LEVEL VALIDATION SET:
  Top-1 Accuracy : 38.1%
  Top-3 Accuracy : 56.9%
  Macro F1       : 35.7%

SONG-LEVEL TEST SET (116 recordings, ~27 segs each):
  Top-1 Accuracy : 57.8%
  Top-3 Accuracy : 75.9%
  Macro F1       : 55.2%

---
Model: Mel CRNN Baseline (baseline_crnn_A)

SONG-LEVEL TEST SET:
  Top-1 Accuracy : 14.7%
  Top-3 Accuracy : 24.1%
  Macro F1       : 12.0%

---
IMPROVEMENT (Swaragram Song vs Mel Song):
  Top-1: +43.1 percentage points (+293%)
  Top-3: +51.8 percentage points (+215%)
"""
    (APP_DIR / "evaluation_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("  [OK] evaluation_summary.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"=== Building Paper Package -> {PAPER_DIR} ===\n")

    copy_existing_figures()
    print()
    fig_dataset_distribution()
    fig_segment_vs_song()
    fig_progression()
    fig_pooling_diagram()
    print()
    generate_tables()
    print()
    assemble_appendix()

    logger.info(f"\n=== Paper package complete ===")
    logger.info(f"\nStructure:")
    for path in sorted(PAPER_DIR.rglob("*")):
        if path.is_file():
            size_kb = path.stat().st_size // 1024
            rel = path.relative_to(PAPER_DIR)
            logger.info(f"  paper/{rel}  ({size_kb} KB)")
