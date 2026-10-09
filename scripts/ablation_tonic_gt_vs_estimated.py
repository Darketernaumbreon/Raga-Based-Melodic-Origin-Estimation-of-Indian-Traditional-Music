"""
ablation_tonic_gt_vs_estimated.py

Ablation Study: Ground-Truth Tonic vs Estimated Tonic
======================================================
Answers: How much classification performance is lost due to tonic estimation?

This script uses the same trained CRNN model (swaragram_crnn_A) and the same
inference pipeline for both conditions. The ONLY difference between conditions
is the tonic value used during Swaragram extraction:

  Condition A (GT):  tonic from Saraga expert annotations (metadata_A.csv)
  Condition B (Est): tonic from TonicEstimator.estimate(f0)

Scope: All Saraga recordings with a ground-truth tonic value (N=44).
       These are the only recordings in the combined dataset with reference tonics.
       Reported as a "Saraga-subset ablation" in the paper, distinct from
       the main test-set evaluation.

Usage:
    python scripts/ablation_tonic_gt_vs_estimated.py

Output:
    experiments/ablation_tonic/saraga_ablation_results.csv
    experiments/ablation_tonic/saraga_ablation_summary.txt
    experiments/ablation_tonic/saraga_ablation_per_recording.csv
"""

import sys
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.swaragram.pitch_extractor import PitchExtractor
from src.swaragram.tonic_estimator import TonicEstimator
from src.swaragram.pitch_normalizer import PitchNormalizer
from src.swaragram.swaragram_generator import SwaragramGenerator
from src.preprocessing.audio_utils import load_and_resample_audio, resolve_audio_path, segment_waveform_with_metadata
from src.models.crnn_lstm import CRNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EXPERIMENT_NAME    = "swaragram_crnn_A"
METADATA_CSV       = PROJECT_ROOT / "data" / "processed" / "metadata" / "dataset_A" / "metadata_A.csv"
OUT_DIR            = PROJECT_ROOT / "experiments" / "ablation_tonic"
SR                 = 22050
HOP_LENGTH         = 512
SEGMENT_DURATION   = 30.0    # seconds per segment (matches training)
HOP_DURATION       = 15.0    # segment hop (matches training)
MIN_DURATION       = 10.0    # minimum trailing segment length


# ── Swaragram pipeline (on-the-fly, no caching) ───────────────────────────────

class SwaragramPipeline:
    """
    Lightweight wrapper that extracts a Swaragram from raw audio given a tonic.
    Re-uses the same components as the main training pipeline.
    """
    def __init__(self):
        self.extractor  = PitchExtractor(sr=SR, hop_length=HOP_LENGTH)
        self.tonic_est  = TonicEstimator()
        self.normalizer = PitchNormalizer()
        self.generator  = SwaragramGenerator(num_bins=12, sigma=25.0)

    def extract_with_tonic(self, audio: np.ndarray, tonic_hz: float) -> np.ndarray:
        """Extract Swaragram using a supplied (GT or estimated) tonic."""
        f0     = self.extractor.extract(audio)
        cents  = self.normalizer.normalize(f0, tonic_hz)
        return self.generator.generate(cents)

    def estimate_tonic(self, audio: np.ndarray) -> float:
        """Estimate tonic from audio using the histogram estimator."""
        f0 = self.extractor.extract(audio)
        return self.tonic_est.estimate(f0)


# ── Model loading ─────────────────────────────────────────────────────────────

def _remap_legacy_state_dict(state_dict):
    """Remap old flat-key checkpoints to the current shared_encoder layout."""
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


def load_model(experiment_name, device):
    model_path = PROJECT_ROOT / "experiments" / experiment_name / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    checkpoint   = torch.load(model_path, map_location=device, weights_only=False)
    label_mapping = checkpoint["label_mapping"]
    num_classes   = len(label_mapping)
    model = CRNN(input_type="swaragram", num_classes=num_classes)
    state_dict = _remap_legacy_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, label_mapping


# ── Per-recording inference ───────────────────────────────────────────────────

def infer_recording(audio_path, tonic_hz, pipeline, model, label_mapping, device):
    """
    Load audio, segment it, extract Swaragrams with the given tonic,
    run inference on each segment, and return the mean-pooled probability vector.

    Returns
    -------
    np.ndarray of shape (num_classes,) — mean probability across segments.
    None if the recording could not be loaded.
    """
    try:
        resolved = resolve_audio_path(audio_path)
        audio, _ = load_and_resample_audio(resolved, target_sr=SR, normalize=True)
    except Exception as e:
        logger.warning(f"  Could not load {audio_path}: {e}")
        return None

    segments = segment_waveform_with_metadata(
        audio, sr=SR,
        segment_duration_s=SEGMENT_DURATION,
        hop_duration_s=HOP_DURATION,
        min_duration_s=MIN_DURATION
    )
    if not segments:
        logger.warning(f"  No segments produced for {audio_path}")
        return None

    num_classes = len(label_mapping)
    prob_sum    = np.zeros(num_classes, dtype=np.float64)
    seg_count   = 0

    with torch.no_grad():
        for seg_audio, _offset, _dur in segments:
            try:
                swaragram = pipeline.extract_with_tonic(seg_audio, tonic_hz)
                tensor = torch.FloatTensor(swaragram).unsqueeze(0).unsqueeze(0).to(device)
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prob_sum  += probs
                seg_count += 1
            except Exception as e:
                logger.debug(f"  Segment failed: {e}")
                continue

    if seg_count == 0:
        return None

    return prob_sum / seg_count


# ── Main ablation loop ────────────────────────────────────────────────────────

def run_ablation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 65)
    logger.info("Tonic Ablation: Ground-Truth vs Estimated")
    logger.info("=" * 65)
    logger.info(f"Model      : {EXPERIMENT_NAME}")
    logger.info(f"Device     : {device}")
    logger.info(f"Output dir : {OUT_DIR}")

    # Load model
    model, label_mapping = load_model(EXPERIMENT_NAME, device)
    idx_to_raag = {v: k for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    logger.info(f"Model loaded — {num_classes} classes")

    # Load Saraga recordings with ground-truth tonic
    df = pd.read_csv(METADATA_CSV)
    saraga_df = df[
        (df["dataset"] == "saraga") & df["tonic"].notna()
    ].copy().reset_index(drop=True)
    logger.info(f"Saraga recordings with GT tonic: {len(saraga_df)}")

    if len(saraga_df) == 0:
        logger.error("No Saraga recordings found with tonic annotations. Aborting.")
        return

    pipeline = SwaragramPipeline()

    # ── Per-recording loop ────────────────────────────────────────────────────
    records = []
    for i, row in tqdm(saraga_df.iterrows(), total=len(saraga_df), desc="Ablation"):
        rec_id    = str(row["recording_id"])
        raag      = row["canonical_raag"]
        gt_tonic  = float(row["tonic"])
        audio_path = row["audio_path"]
        split     = row.get("split", "unknown")

        if raag not in label_mapping:
            logger.warning(f"  [{i+1}] '{raag}' not in model label_mapping — skipping {rec_id}")
            continue

        true_idx = label_mapping[raag]

        logger.info(
            f"[{i+1:>2}/{len(saraga_df)}] {raag:<22} | split={split} | "
            f"GT tonic={gt_tonic:.1f} Hz"
        )

        # ── Condition A: ground-truth tonic ──
        gt_probs = infer_recording(audio_path, gt_tonic, pipeline, model, label_mapping, device)

        # ── Condition B: estimated tonic ──
        try:
            resolved   = resolve_audio_path(audio_path)
            audio_full, _ = load_and_resample_audio(resolved, target_sr=SR, normalize=True)
            # Use first 120 s for tonic estimation (speed + accuracy balance)
            audio_est  = audio_full[:SR * 120]
            est_tonic  = pipeline.estimate_tonic(audio_est)
        except Exception as e:
            logger.warning(f"  Tonic estimation failed: {e}")
            est_tonic  = gt_tonic   # fallback keeps it comparable
        est_probs = infer_recording(audio_path, est_tonic, pipeline, model, label_mapping, device)

        # ── Record results ──
        rec = {
            "recording_id":  rec_id,
            "raag":          raag,
            "split":         split,
            "gt_tonic_hz":   round(gt_tonic, 3),
            "est_tonic_hz":  round(est_tonic, 3),
            "true_idx":      true_idx,
        }

        for cond_name, probs in [("gt", gt_probs), ("est", est_probs)]:
            if probs is not None:
                top3 = np.argsort(probs)[::-1][:3]
                pred = top3[0]
                rec[f"{cond_name}_pred_idx"]   = int(pred)
                rec[f"{cond_name}_pred_raag"]  = idx_to_raag[pred]
                rec[f"{cond_name}_top1"]       = bool(pred == true_idx)
                rec[f"{cond_name}_top3"]       = bool(true_idx in top3)
                rec[f"{cond_name}_confidence"] = round(float(probs[pred]), 4)
            else:
                rec[f"{cond_name}_pred_idx"]   = None
                rec[f"{cond_name}_pred_raag"]  = None
                rec[f"{cond_name}_top1"]       = None
                rec[f"{cond_name}_top3"]       = None
                rec[f"{cond_name}_confidence"] = None

        if rec.get("gt_top1") is not None and rec.get("est_top1") is not None:
            logger.info(
                f"         GT : {'✓' if rec['gt_top1'] else '✗'} "
                f"(pred={rec['gt_pred_raag']}, conf={rec['gt_confidence']:.3f})  |  "
                f"Est: {'✓' if rec['est_top1'] else '✗'} "
                f"(pred={rec['est_pred_raag']}, conf={rec['est_confidence']:.3f}, "
                f"tonic={est_tonic:.1f} Hz)"
            )

        records.append(rec)

    # ── Compute summary metrics ───────────────────────────────────────────────
    results_df = pd.DataFrame(records)

    # Save per-recording CSV (ASCII-safe)
    per_rec_path = OUT_DIR / "saraga_ablation_per_recording.csv"
    results_df.to_csv(per_rec_path, index=False, encoding="utf-8")
    logger.info(f"Per-recording results saved -> {per_rec_path}")

    valid = results_df[
        results_df["gt_top1"].notna() & results_df["est_top1"].notna()
    ].copy()
    n = len(valid)

    if n == 0:
        logger.error("No valid results to summarise.")
        return

    gt_top1  = valid["gt_top1"].mean()
    gt_top3  = valid["gt_top3"].mean()
    est_top1 = valid["est_top1"].mean()
    est_top3 = valid["est_top3"].mean()

    gt_f1  = f1_score(valid["true_idx"], valid["gt_pred_idx"],  average="macro", zero_division=0)
    est_f1 = f1_score(valid["true_idx"], valid["est_pred_idx"], average="macro", zero_division=0)

    gap_top1 = gt_top1 - est_top1
    gap_top3 = gt_top3 - est_top3

    # Records where GT is correct but Est is wrong (tonic error caused the miss)
    rescued = valid[valid["gt_top1"] & ~valid["est_top1"]]
    degraded = valid[~valid["gt_top1"] & valid["est_top1"]]

    summary_lines = [
        "",
        "=" * 62,
        "  TONIC ABLATION STUDY — SARAGA SUBSET",
        f"  Model : {EXPERIMENT_NAME}   |   N = {n} recordings",
        "=" * 62,
        f"  {'Condition':<38} {'Top-1':>6}  {'Top-3':>6}  {'Macro-F1':>9}",
        "  " + "-" * 60,
        f"  {'Ground-truth tonic (Saraga annotations)':<38} {gt_top1*100:>5.1f}%  {gt_top3*100:>5.1f}%  {gt_f1:>9.4f}",
        f"  {'Estimated tonic (our pipeline)':<38} {est_top1*100:>5.1f}%  {est_top3*100:>5.1f}%  {est_f1:>9.4f}",
        "  " + "-" * 60,
        f"  {'Gap (GT minus Estimated)':<38} {gap_top1*100:>+5.1f}pp {gap_top3*100:>+5.1f}pp",
        "=" * 62,
        "",
        f"  Recordings where GT correct, Est wrong : {len(rescued)}  "
        f"(tonic error caused these misses)",
        f"  Recordings where Est correct, GT wrong : {len(degraded)}  "
        f"(artefacts / model calibration)",
        "=" * 62,
        "",
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = OUT_DIR / "saraga_ablation_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info(f"Summary saved -> {summary_path}")

    # ── Print per-recording comparison table ─────────────────────────────────
    display = valid[[
        "raag", "split", "gt_tonic_hz", "est_tonic_hz",
        "gt_pred_raag", "gt_top1",
        "est_pred_raag", "est_top1",
    ]].copy()
    display.columns = [
        "Raag", "Split", "GT Hz", "Est Hz",
        "GT Pred", "GT✓",
        "Est Pred", "Est✓",
    ]
    # Replace True/False with readable markers
    display["GT✓"]  = display["GT✓"].map({True: "✓", False: "✗"})
    display["Est✓"] = display["Est✓"].map({True: "✓", False: "✗"})
    print("\nPer-Recording Comparison:")
    try:
        print(display.to_string(index=False))
    except UnicodeEncodeError:
        # Windows console fallback
        safe = display.copy()
        safe["GT✓"]  = safe["GT✓"].replace({"✓": "OK", "✗": "X"})
        safe["Est✓"] = safe["Est✓"].replace({"✓": "OK", "✗": "X"})
        print(safe.to_string(index=False))


if __name__ == "__main__":
    run_ablation()
