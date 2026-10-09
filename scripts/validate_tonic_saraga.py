"""
validate_tonic_saraga.py

Quantitative validation of the tonic (Sa) estimator against the
ground-truth tonic annotations provided by the Saraga dataset.

Each Saraga recording includes a .ctonic.txt file containing the
reference tonic frequency in Hz, recorded by expert annotators.

This script:
  1. Crawls the Saraga metadata directory to find all .ctonic.txt files.
  2. Matches each annotation to the corresponding audio file.
  3. Runs the project's PitchExtractor + TonicEstimator pipeline on each recording.
  4. Computes per-recording cent error between estimated and reference tonic.
  5. Reports summary metrics: exact-pitch-class match, within-50-cents,
     within-100-cents, mean absolute cent error, and octave deviation stats.
  6. Saves a detailed CSV, a summary text file, and an error distribution PNG.

Usage:
    python scripts/validate_tonic_saraga.py

Output:
    experiments/tonic_validation/saraga_tonic_validation.csv
    experiments/tonic_validation/saraga_tonic_validation_summary.txt
    experiments/tonic_validation/tonic_error_distribution.png
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.swaragram.pitch_extractor import PitchExtractor
from src.swaragram.tonic_estimator import TonicEstimator
from src.preprocessing.audio_utils import load_and_resample_audio, resolve_audio_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SARAGA_METADATA_DIR = PROJECT_ROOT / "data" / "raw" / "saraga" / "metadata"
SARAGA_AUDIO_DIR    = PROJECT_ROOT / "data" / "raw" / "saraga" / "audio"
OUT_DIR             = PROJECT_ROOT / "experiments" / "tonic_validation"
AUDIO_EXTENSIONS    = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
SR                  = 22050
ANALYSIS_SECONDS    = 120  # Use first 2 minutes; tonic is stable early in a performance


# ── Utility functions ─────────────────────────────────────────────────────────

def hz_to_cents(freq_hz, reference_hz):
    """Signed cent difference: positive = estimated is sharp."""
    if reference_hz <= 0 or freq_hz <= 0:
        return float("nan")
    return 1200.0 * np.log2(freq_hz / reference_hz)


def hz_to_note_name(hz):
    """Convert Hz to nearest pitch-class note name (e.g. C#4)."""
    try:
        midi = librosa.hz_to_midi(float(hz))
        return librosa.midi_to_note(int(round(midi)))
    except Exception:
        return "N/A"


def pitch_class_of(hz):
    """Return pitch-class index 0-11 (C=0) of the nearest semitone."""
    midi = librosa.hz_to_midi(float(hz))
    return int(round(midi)) % 12


def find_audio_file(album_dir, recording_name):
    """
    Locate the audio file for a given recording inside the Saraga audio tree.

    Saraga audio files use a double-extension naming convention, e.g.:
        audio/<album>/<recording>/<recording>.mp3.mp3
    So we glob for ANY audio file inside the recording subfolder rather
    than trying to reconstruct the filename from the stem.
    """
    recording_audio_dir = SARAGA_AUDIO_DIR / album_dir / recording_name
    if recording_audio_dir.is_dir():
        for ext in AUDIO_EXTENSIONS:
            # Direct match inside the recording's own subdirectory
            matches = list(recording_audio_dir.glob(f"*{ext}"))
            if matches:
                return matches[0]
        # Accept any file at all if no known extension matches
        all_files = [f for f in recording_audio_dir.iterdir() if f.is_file()]
        if all_files:
            return all_files[0]

    # Fallback: search recursively under the album folder for a stem match
    album_audio_dir = SARAGA_AUDIO_DIR / album_dir
    if album_audio_dir.is_dir():
        for ext in AUDIO_EXTENSIONS:
            matches = list(album_audio_dir.rglob(f"{recording_name}*{ext}"))
            if matches:
                return matches[0]
    return None


def crawl_saraga_annotations():
    """
    Walk the Saraga metadata tree and collect (ctonic_path, album, recording_name)
    for every .ctonic.txt file found.

    Directory structure assumed:
        metadata/<album>/<recording>/<recording>.ctonic.txt
    """
    entries = []
    for ctonic_path in sorted(SARAGA_METADATA_DIR.rglob("*.ctonic.txt")):
        recording_dir  = ctonic_path.parent
        album_dir      = recording_dir.parent
        recording_name = recording_dir.name   # e.g. "Raag Bhairav"
        album_name     = album_dir.name       # e.g. "Raag Bhairav & Dhani by Kumar Gandharva"
        entries.append({
            "ctonic_path":    ctonic_path,
            "album":          album_name,
            "recording_name": recording_name,
        })
    return entries


def read_ctonic(ctonic_path):
    """Read the reference tonic Hz value from a Saraga .ctonic.txt file."""
    try:
        content = ctonic_path.read_text(encoding="utf-8").strip()
        return float(content)
    except Exception as e:
        logger.warning(f"Could not read {ctonic_path}: {e}")
        return None


# ── Per-recording evaluation ──────────────────────────────────────────────────

def evaluate_one(entry, extractor, tonic_est):
    """Run the pipeline on one recording and return a result dict."""
    album          = entry["album"]
    recording_name = entry["recording_name"]
    ctonic_path    = entry["ctonic_path"]

    ref_tonic_hz = read_ctonic(ctonic_path)
    audio_file   = find_audio_file(album, recording_name)

    result = {
        "album":          album,
        "recording":      recording_name,
        "ref_tonic_hz":   round(ref_tonic_hz, 3) if ref_tonic_hz else None,
        "ref_note":       hz_to_note_name(ref_tonic_hz) if ref_tonic_hz else "N/A",
        "est_tonic_hz":   None,
        "est_note":       "N/A",
        "cent_error":     None,
        "abs_cent_error": None,
        "octave_error":   None,
        "pc_match":       None,
        "within_50c":     None,
        "within_100c":    None,
        "status":         "PENDING",
    }

    if ref_tonic_hz is None:
        result["status"] = "NO_REFERENCE"
        return result

    if audio_file is None:
        result["status"] = "NO_AUDIO"
        return result

    try:
        audio, _ = load_and_resample_audio(
            audio_file,
            target_sr=SR,
            duration=float(ANALYSIS_SECONDS),
            normalize=True,
        )
        f0 = extractor.extract(audio)
        est_hz = tonic_est.estimate(f0)

        cent_err     = hz_to_cents(est_hz, ref_tonic_hz)
        abs_cent_err = abs(cent_err)

        # Octave error: estimated is roughly one octave off from reference
        octave_err = (abs(abs_cent_err - 1200) < 100) or (abs(abs_cent_err - 2400) < 100)

        # Pitch-class match: same semitone mod 12 (octave-invariant)
        pc_match = pitch_class_of(est_hz) == pitch_class_of(ref_tonic_hz)

        result.update({
            "est_tonic_hz":   round(est_hz, 3),
            "est_note":       hz_to_note_name(est_hz),
            "cent_error":     round(cent_err, 1),
            "abs_cent_error": round(abs_cent_err, 1),
            "octave_error":   bool(octave_err),
            "pc_match":       bool(pc_match),
            "within_50c":     bool(abs_cent_err <= 50.0),
            "within_100c":    bool(abs_cent_err <= 100.0),
            "status":         "OK",
        })

    except Exception as e:
        logger.warning(f"  Pipeline failed for '{recording_name}': {e}")
        result["status"] = f"ERROR: {e}"

    return result


# ── Summary metrics ───────────────────────────────────────────────────────────

def compute_summary(df):
    ok = df[df["status"] == "OK"].copy()
    n_total = len(df)
    n_ok    = len(ok)
    n_skip  = len(df[df["status"].isin(["NO_AUDIO", "NO_REFERENCE"])])
    n_error = len(df[~df["status"].isin(["OK", "NO_AUDIO", "NO_REFERENCE", "PENDING"])])

    if n_ok == 0:
        return {"n_total": n_total, "n_ok": n_ok, "note": "No successful evaluations."}

    pc_match_pct   = 100.0 * ok["pc_match"].sum()    / n_ok
    within_50_pct  = 100.0 * ok["within_50c"].sum()  / n_ok
    within_100_pct = 100.0 * ok["within_100c"].sum() / n_ok
    mean_ace       = ok["abs_cent_error"].mean()
    median_ace     = ok["abs_cent_error"].median()
    octave_pct     = 100.0 * ok["octave_error"].sum() / n_ok

    return {
        "Total recordings":           n_total,
        "Successfully evaluated":     n_ok,
        "Skipped (no audio / ref)":   n_skip,
        "Errors":                     n_error,
        "Pitch-class match (%)":      round(pc_match_pct,   1),
        "Within +/-50 cents (%)":     round(within_50_pct,  1),
        "Within +/-100 cents (%)":    round(within_100_pct, 1),
        "Mean absolute cent error":   round(mean_ace,       1),
        "Median absolute cent error": round(median_ace,     1),
        "Octave errors (%)":          round(octave_pct,     1),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_error_distribution_plot(df, out_dir):
    ok = df[df["status"] == "OK"].copy()
    if len(ok) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Tonic Estimation Error vs Saraga Ground Truth",
        fontsize=14, fontweight="bold"
    )

    # Absolute cent error distribution
    axes[0].hist(
        ok["abs_cent_error"], bins=20,
        color="#4C72B0", edgecolor="white", alpha=0.85
    )
    axes[0].axvline(50,  color="orange", linestyle="--", linewidth=1.5, label="+/-50 cents")
    axes[0].axvline(100, color="red",    linestyle="--", linewidth=1.5, label="+/-100 cents")
    axes[0].set_xlabel("Absolute Cent Error", fontsize=12)
    axes[0].set_ylabel("Number of Recordings", fontsize=12)
    axes[0].set_title("Absolute Cent Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Signed cent error (shows bias: sharp vs flat)
    axes[1].hist(
        ok["cent_error"], bins=20,
        color="#DD8452", edgecolor="white", alpha=0.85
    )
    axes[1].axvline(0, color="black", linestyle="-", linewidth=1.5, label="0 cents (perfect)")
    axes[1].set_xlabel("Signed Cent Error (estimated - reference)", fontsize=12)
    axes[1].set_ylabel("Number of Recordings", fontsize=12)
    axes[1].set_title("Signed Cent Error (Bias Check)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "tonic_error_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error distribution plot: {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("Saraga Tonic Estimation Validation")
    logger.info("=" * 65)
    logger.info(f"Metadata root : {SARAGA_METADATA_DIR}")
    logger.info(f"Audio root    : {SARAGA_AUDIO_DIR}")
    logger.info(f"Output dir    : {OUT_DIR}")
    logger.info(f"Analysis window : first {ANALYSIS_SECONDS}s per recording")

    annotations = crawl_saraga_annotations()
    logger.info(f"Found {len(annotations)} .ctonic.txt annotations")

    if not annotations:
        logger.error("No .ctonic.txt files found. Check SARAGA_METADATA_DIR path.")
        return

    extractor = PitchExtractor(sr=SR, hop_length=512)
    tonic_est = TonicEstimator()

    results = []
    for i, entry in enumerate(annotations):
        logger.info(
            f"[{i+1:>3}/{len(annotations)}]  "
            f"{entry['album'][:42]:<42} | {entry['recording_name']}"
        )
        result = evaluate_one(entry, extractor, tonic_est)
        results.append(result)

        if result["status"] == "OK":
            sign = "+" if result["cent_error"] >= 0 else ""
            logger.info(
                f"          Ref: {result['ref_tonic_hz']:>8.2f} Hz ({result['ref_note']:<5})  "
                f"Est: {result['est_tonic_hz']:>8.2f} Hz ({result['est_note']:<5})  "
                f"Error: {sign}{result['cent_error']:>7.1f} c  "
                f"PC: {'OK' if result['pc_match'] else 'MISS'}"
            )
        else:
            logger.info(f"          Status: {result['status']}")

    # Save detailed CSV
    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "saraga_tonic_validation.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"\nDetailed results saved -> {csv_path}")

    # Compute and display summary
    summary = compute_summary(df)
    summary_lines = [
        "",
        "=" * 55,
        "  TONIC ESTIMATION VALIDATION SUMMARY",
        "  (Against Saraga Expert Annotations)",
        "=" * 55,
    ]
    for k, v in summary.items():
        summary_lines.append(f"  {k:<35} {v}")
    summary_lines += ["=" * 55, ""]
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = OUT_DIR / "saraga_tonic_validation_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info(f"Summary saved -> {summary_path}")

    # Error distribution plots
    save_error_distribution_plot(df, OUT_DIR)

    # Print per-recording table for OK results
    ok_df = df[df["status"] == "OK"][[
        "recording", "ref_note", "est_note", "cent_error",
        "abs_cent_error", "pc_match", "within_50c", "octave_error"
    ]].copy()
    ok_df.columns = [
        "Recording", "Ref Sa", "Est Sa", "Cent Err",
        "|Cent Err|", "PC Match", "<=50c", "Octave Err"
    ]
    # Sanitise sharp/flat symbols for Windows console (cp1252 cannot encode ♯)
    ok_df = ok_df.applymap(
        lambda x: x.replace("\u266f", "#").replace("\u266d", "b") if isinstance(x, str) else x
    )
    print("\nPer-Recording Results:")
    print(ok_df.to_string(index=False))

    # Warn if quality is low
    ok_within100 = summary.get("Within +/-100 cents (%)", 0)
    if isinstance(ok_within100, (int, float)) and ok_within100 < 50:
        logger.warning(
            "CAUTION: Fewer than 50% of recordings within +/-100 cents. "
            "Tonic estimator may be unreliable -- review generated Swaragrams carefully."
        )


if __name__ == "__main__":
    main()
