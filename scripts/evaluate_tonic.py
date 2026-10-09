"""
evaluate_tonic.py

Qualitative sanity-check for the tonic (Sa) estimator.
Samples 25 recordings from the dataset, runs the full pitch extraction
and tonic estimation pipeline, and produces:
  1. A console report with estimated Sa in Hz and the nearest musical note.
  2. A saved pitch histogram PNG per recording (in experiments/tonic_eval/).

This script validates the preprocessing section of the paper.
"""
import sys
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.swaragram.pitch_extractor import PitchExtractor
from src.swaragram.tonic_estimator import TonicEstimator
from src.preprocessing.audio_utils import load_and_resample_audio, resolve_audio_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def hz_to_note_name(hz):
    """Convert Hz to nearest musical note name."""
    try:
        midi = librosa.hz_to_midi(hz)
        note = librosa.midi_to_note(round(midi))
        return note
    except Exception:
        return "N/A"


def evaluate_sample(rec, extractor, tonic_est, out_dir, sr=22050):
    """Process one recording and save its pitch histogram."""
    rec_id = rec["recording_id"]
    audio_path = rec["audio_path"]

    try:
        resolved = resolve_audio_path(audio_path)
        audio, _ = load_and_resample_audio(resolved, target_sr=sr, normalize=True)
        
        # Use only the first 60s for speed (tonic is usually stable early in the performance)
        audio = audio[:sr * 60]
        
        f0 = extractor.extract(audio)
        tonic_hz = tonic_est.estimate(f0)
        note_name = hz_to_note_name(tonic_hz)

        # Build and save pitch histogram
        f0_valid = f0[(~np.isnan(f0)) & (f0 > 0)]
        if len(f0_valid) > 0:
            midi_vals = librosa.hz_to_midi(f0_valid)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(midi_vals, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
            ax.axvline(x=librosa.hz_to_midi(tonic_hz), color='red', linewidth=2,
                       label=f"Estimated Sa = {tonic_hz:.1f} Hz ({note_name})")
            ax.set_title(f"Pitch Histogram | Recording: {rec_id}")
            ax.set_xlabel("MIDI Note")
            ax.set_ylabel("Frame Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"{rec_id}_tonic.png", dpi=150)
            plt.close()

        return {
            "recording_id": rec_id,
            "audio_path": audio_path,
            "tonic_hz": round(tonic_hz, 2),
            "tonic_note": note_name,
            "status": "OK"
        }

    except Exception as e:
        logger.warning(f"Failed: {rec_id} — {e}")
        return {
            "recording_id": rec_id,
            "audio_path": audio_path,
            "tonic_hz": None,
            "tonic_note": None,
            "status": f"ERROR: {e}"
        }


def main():
    sr = 22050
    n_samples = 25
    random.seed(42)

    out_dir = PROJECT_ROOT / "experiments" / "tonic_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Load dataset splits and sample recordings
    splits_dir = PROJECT_ROOT / "data" / "processed" / "metadata"
    all_csvs = list(splits_dir.rglob("*.csv"))
    
    if not all_csvs:
        logger.error("No split CSVs found. Check data/processed/metadata/")
        return

    df = pd.concat([pd.read_csv(p) for p in all_csvs], ignore_index=True)
    df = df.dropna(subset=["audio_path"]).drop_duplicates(subset=["recording_id"])
    
    sample = df.sample(min(n_samples, len(df)), random_state=42).to_dict("records")
    logger.info(f"Evaluating tonic estimation on {len(sample)} recordings...")

    extractor = PitchExtractor(sr=sr, hop_length=512)
    tonic_est = TonicEstimator()

    results = []
    for i, rec in enumerate(sample):
        logger.info(f"[{i+1}/{len(sample)}] Processing: {rec['recording_id']}")
        result = evaluate_sample(rec, extractor, tonic_est, out_dir, sr=sr)
        results.append(result)
        logger.info(f"  → Estimated Sa: {result['tonic_hz']} Hz  ({result['tonic_note']})")

    results_df = pd.DataFrame(results)
    csv_path = out_dir / "tonic_evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)

    logger.info(f"\n=== Tonic Estimation Complete ===")
    logger.info(f"Results saved to {csv_path}")
    logger.info(f"Pitch histograms saved to {out_dir}/")
    logger.info(f"\nSummary:")
    logger.info(f"  OK:     {(results_df['status'] == 'OK').sum()}/{len(results_df)}")
    logger.info(f"  Errors: {(results_df['status'] != 'OK').sum()}/{len(results_df)}")
    logger.info(f"\nSample Results:")
    print(results_df[["recording_id", "tonic_hz", "tonic_note", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
