"""
generate_fig8_ci.py

Generates Figure 8: 95% Bootstrap Confidence Interval Grouped Bar Chart with Error Bars
Comparing Mel-CRNN vs. Swaragram-CRNN across Top-1, Top-3, and Macro F1.
Saved to: experiments/paper_figures/fig8_bootstrap_ci.png
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load stats JSON
stats_file = PROJECT_ROOT / "experiments" / "paper_stats" / "statistical_significance_results.json"
with open(stats_file) as f:
    data = json.load(f)

ci = data["confidence_intervals"]

metrics = ["Top-1 Accuracy", "Top-3 Accuracy", "Macro F1"]

# Means
mel_means = [ci["mel_top1"]["mean"] * 100, ci["mel_top3"]["mean"] * 100, ci["mel_f1"]["mean"] * 100]
swara_means = [ci["swara_top1"]["mean"] * 100, ci["swara_top3"]["mean"] * 100, ci["swara_f1"]["mean"] * 100]

# Error bar calculations (symmetric/asymmetric delta from mean)
mel_err_low = [mel_means[0] - ci["mel_top1"]["ci_95_low"]*100, mel_means[1] - ci["mel_top3"]["ci_95_low"]*100, mel_means[2] - ci["mel_f1"]["ci_95_low"]*100]
mel_err_high = [ci["mel_top1"]["ci_95_high"]*100 - mel_means[0], ci["mel_top3"]["ci_95_high"]*100 - mel_means[1], ci["mel_f1"]["ci_95_high"]*100 - mel_means[2]]
mel_yerr = [mel_err_low, mel_err_high]

swara_err_low = [swara_means[0] - ci["swara_top1"]["ci_95_low"]*100, swara_means[1] - ci["swara_top3"]["ci_95_low"]*100, swara_means[2] - ci["swara_f1"]["ci_95_low"]*100]
swara_err_high = [ci["swara_top1"]["ci_95_high"]*100 - swara_means[0], ci["swara_top3"]["ci_95_high"]*100 - swara_means[1], ci["swara_f1"]["ci_95_high"]*100 - swara_means[2]]
swara_yerr = [swara_err_low, swara_err_high]

x = np.arange(len(metrics))
width = 0.35

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
})

fig, ax = plt.subplots(figsize=(8.5, 4.8))

rects1 = ax.bar(x - width/2, mel_means, width, yerr=mel_yerr, capsize=5,
                label='Baseline Mel-CRNN', color='#E63946', alpha=0.85, edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + width/2, swara_means, width, yerr=swara_yerr, capsize=5,
                label='Proposed Swaragram-CRNN', color='#2A9D8F', alpha=0.9, edgecolor='black', linewidth=0.8)

# Annotate difference / significance
ax.text(0, 70, r'$\Delta = +43.0\%\ [p = 9.07 \times 10^{-10}]$', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4F8", edgecolor="#2A9D8F", lw=1.2))
ax.text(1, 87, r'$\Delta = +51.7\%\ [p < 0.001]$', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4F8", edgecolor="#2A9D8F", lw=1.2))
ax.text(2, 63, r'$\Delta = +0.399\ [p < 0.001]$', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4F8", edgecolor="#2A9D8F", lw=1.2))

# Value labels on top of bars
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h / 2),
                xytext=(0, 0), textcoords="offset points",
                ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h / 2),
                xytext=(0, 0), textcoords="offset points",
                ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')

ax.set_ylabel('Score (%) / Index (0-100)', fontsize=11, fontweight='bold')
ax.set_title('Figure 8: 95% Bootstrap Confidence Intervals (N = 10,000 Resamples)\nMel-CRNN vs. Swaragram-CRNN on Test Set (N = 116)',
             fontsize=11.5, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
fig_path = OUT_DIR / "fig8_bootstrap_ci.png"
fig.savefig(fig_path)
plt.close(fig)
print(f"Successfully generated Figure 8 at: {fig_path}")
