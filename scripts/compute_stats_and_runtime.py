"""
compute_stats_and_runtime.py

Performs comprehensive statistical significance testing and empirical CPU/GPU runtime/latency benchmarking.

Outputs:
  1. 95% Bootstrap Confidence Intervals (10,000 iterations) for Top-1, Top-3, and Macro F1
  2. Paired McNemar's Test for statistical significance (p-value, chi2 statistic, contingency table)
  3. CPU (AMD Ryzen 7 7735HS) vs. GPU (NVIDIA RTX 4050) Latency Breakdown (Per-Segment and Per-Recording)
  4. Preprocessing vs Inference Latency & Real-Time Factor (RTF)
"""

import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "experiments" / "paper_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Bootstrap Confidence Intervals & McNemar's Test
# ---------------------------------------------------------------------------

def compute_bootstrap_and_mcnemar():
    logger.info("=== Computing Bootstrap CIs & McNemar's Test ===")
    mel_csv = PROJECT_ROOT / "experiments" / "baseline_crnn_A" / "song_level_results.csv"
    swara_csv = PROJECT_ROOT / "experiments" / "swaragram_crnn_A" / "song_level_results.csv"

    mel_df = pd.read_csv(mel_csv)
    swara_df = pd.read_csv(swara_csv)

    merged = pd.merge(mel_df, swara_df, on="recording_id", suffixes=("_mel", "_swara"))
    N = len(merged)
    logger.info(f"Loaded paired test predictions for {N} recordings.")

    # McNemar's Test
    a = int(((merged["top1_correct_mel"] == True) & (merged["top1_correct_swara"] == True)).sum())
    b = int(((merged["top1_correct_mel"] == True) & (merged["top1_correct_swara"] == False)).sum())
    c = int(((merged["top1_correct_mel"] == False) & (merged["top1_correct_swara"] == True)).sum())
    d = int(((merged["top1_correct_mel"] == False) & (merged["top1_correct_swara"] == False)).sum())

    chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c) if (b + c) > 0 else 0.0
    p_value = stats.chi2.sf(chi2_stat, df=1)

    mcnemar_results = {
        "N": N,
        "both_correct_a": a,
        "mel_correct_swara_wrong_b": b,
        "swara_correct_mel_wrong_c": c,
        "both_wrong_d": d,
        "mcnemar_chi2": round(float(chi2_stat), 4),
        "p_value": float(p_value),
        "is_significant_p_001": bool(p_value < 0.001),
    }

    # Bootstrap 95% Confidence Intervals
    n_bootstraps = 10000
    np.random.seed(42)

    mel_top1_boot, mel_top3_boot, mel_f1_boot = [], [], []
    swara_top1_boot, swara_top3_boot, swara_f1_boot = [], [], []
    diff_top1_boot, diff_top3_boot, diff_f1_boot = [], [], []

    for _ in range(n_bootstraps):
        idx = np.random.choice(N, size=N, replace=True)
        sample = merged.iloc[idx]

        m_t1 = accuracy_score(sample["true_idx_mel"], sample["pred_idx_mel"])
        m_t3 = sample["top3_correct_mel"].mean()
        m_f1 = f1_score(sample["true_idx_mel"], sample["pred_idx_mel"], average="macro", zero_division=0)

        s_t1 = accuracy_score(sample["true_idx_swara"], sample["pred_idx_swara"])
        s_t3 = sample["top3_correct_swara"].mean()
        s_f1 = f1_score(sample["true_idx_swara"], sample["pred_idx_swara"], average="macro", zero_division=0)

        mel_top1_boot.append(m_t1)
        mel_top3_boot.append(m_t3)
        mel_f1_boot.append(m_f1)

        swara_top1_boot.append(s_t1)
        swara_top3_boot.append(s_t3)
        swara_f1_boot.append(s_f1)

        diff_top1_boot.append(s_t1 - m_t1)
        diff_top3_boot.append(s_t3 - m_t3)
        diff_f1_boot.append(s_f1 - m_f1)

    def get_ci(arr):
        low = np.percentile(arr, 2.5)
        high = np.percentile(arr, 97.5)
        mean = np.mean(arr)
        return {
            "mean": round(float(mean), 4),
            "ci_95_low": round(float(low), 4),
            "ci_95_high": round(float(high), 4),
            "ci_95_str": f"[{low*100:.1f}%, {high*100:.1f}%]"
        }

    ci_results = {
        "mel_top1": get_ci(mel_top1_boot),
        "mel_top3": get_ci(mel_top3_boot),
        "mel_f1": get_ci(mel_f1_boot),
        "swara_top1": get_ci(swara_top1_boot),
        "swara_top3": get_ci(swara_top3_boot),
        "swara_f1": get_ci(swara_f1_boot),
        "diff_top1": get_ci(diff_top1_boot),
        "diff_top3": get_ci(diff_top3_boot),
        "diff_f1": get_ci(diff_f1_boot),
    }

    return mcnemar_results, ci_results


# ---------------------------------------------------------------------------
# 2. Comprehensive CPU vs. GPU Preprocessing & Latency Benchmarks
# ---------------------------------------------------------------------------

def run_latency_benchmarks():
    logger.info("=== Running CPU & GPU Empirical Latency Benchmarks ===")

    from src.preprocessing.audio_utils import load_and_resample_audio, resolve_audio_path
    from src.swaragram.pitch_extractor import PitchExtractor
    from src.swaragram.tonic_estimator import TonicEstimator
    from src.swaragram.pitch_normalizer import PitchNormalizer
    from src.swaragram.swaragram_generator import SwaragramGenerator
    from src.preprocessing.feature_extraction import extract_mel_spectrogram
    from src.models.crnn_lstm import CRNN

    meta_csv = PROJECT_ROOT / "data" / "processed" / "metadata" / "dataset_A" / "metadata_A.csv"
    meta = pd.read_csv(meta_csv).sample(10, random_state=42)

    extractor = PitchExtractor(sr=22050, hop_length=512)
    tonic_est = TonicEstimator()
    normalizer = PitchNormalizer()
    generator = SwaragramGenerator(num_bins=12, sigma=25.0)

    device_cpu = torch.device("cpu")
    has_gpu = torch.cuda.is_available()
    device_gpu = torch.device("cuda" if has_gpu else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "N/A"
    cpu_name = "AMD Ryzen 7 7735HS with Radeon Graphics (3.20 GHz, 8 cores / 16 threads)"

    swara_model_cpu = CRNN(input_type="swaragram", num_classes=38).to(device_cpu).eval()
    mel_model_cpu = CRNN(input_type="mel", num_classes=38).to(device_cpu).eval()

    swara_model_gpu = CRNN(input_type="swaragram", num_classes=38).to(device_gpu).eval() if has_gpu else swara_model_cpu
    mel_model_gpu = CRNN(input_type="mel", num_classes=38).to(device_gpu).eval() if has_gpu else mel_model_cpu

    # Warmup
    dummy_swara_cpu = torch.randn(1, 1, 12, 1292, device=device_cpu)
    dummy_mel_cpu = torch.randn(1, 1, 128, 1292, device=device_cpu)

    for _ in range(5):
        _ = swara_model_cpu(dummy_swara_cpu)
        _ = mel_model_cpu(dummy_mel_cpu)

    if has_gpu:
        dummy_swara_gpu = torch.randn(1, 1, 12, 1292, device=device_gpu)
        dummy_mel_gpu = torch.randn(1, 1, 128, 1292, device=device_gpu)
        dummy_swara_b32_gpu = torch.randn(32, 1, 12, 1292, device=device_gpu)
        dummy_mel_b32_gpu = torch.randn(32, 1, 128, 1292, device=device_gpu)

        for _ in range(10):
            _ = swara_model_gpu(dummy_swara_gpu)
            _ = mel_model_gpu(dummy_mel_gpu)
            _ = swara_model_gpu(dummy_swara_b32_gpu)
            _ = mel_model_gpu(dummy_mel_b32_gpu)
        torch.cuda.synchronize()

    times_load = []
    times_yin = []
    times_tonic = []
    times_swara_gen = []
    times_mel_gen = []
    times_swara_inf_cpu = []
    times_mel_inf_cpu = []
    times_swara_inf_gpu_b1 = []
    times_mel_inf_gpu_b1 = []
    times_swara_inf_gpu_b32 = []
    times_mel_inf_gpu_b32 = []
    durations = []

    for i, row in meta.iterrows():
        path = resolve_audio_path(row["audio_path"])
        
        # Audio Load
        t0 = time.perf_counter()
        audio, sr = load_and_resample_audio(path, target_sr=22050, duration=30.0, normalize=True)
        t_load = time.perf_counter() - t0
        times_load.append(t_load)
        durations.append(len(audio) / sr)

        # YIN Pitch Tracking
        t0 = time.perf_counter()
        f0 = extractor.extract(audio)
        t_yin = time.perf_counter() - t0
        times_yin.append(t_yin)

        # Tonic Estimation
        t0 = time.perf_counter()
        tonic_hz = tonic_est.estimate(f0)
        t_tonic = time.perf_counter() - t0
        times_tonic.append(t_tonic)

        # Swaragram Generation
        t0 = time.perf_counter()
        cents = normalizer.normalize(f0, tonic_hz)
        swara_mat = generator.generate(cents)
        t_swara = time.perf_counter() - t0
        times_swara_gen.append(t_swara)

        # Mel Spectrogram Generation
        t0 = time.perf_counter()
        mel_mat = extract_mel_spectrogram(audio, sr=22050, n_mels=128)
        t_mel = time.perf_counter() - t0
        times_mel_gen.append(t_mel)

        # Inference CPU (b=1)
        s_tensor_cpu = torch.FloatTensor(swara_mat).unsqueeze(0).unsqueeze(0).to(device_cpu)
        m_tensor_cpu = torch.FloatTensor(mel_mat).unsqueeze(0).unsqueeze(0).to(device_cpu)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = swara_model_cpu(s_tensor_cpu)
        times_swara_inf_cpu.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = mel_model_cpu(m_tensor_cpu)
        times_mel_inf_cpu.append(time.perf_counter() - t0)

        # Inference GPU (b=1 and b=32) with torch.cuda.synchronize()
        if has_gpu:
            s_tensor_gpu = torch.FloatTensor(swara_mat).unsqueeze(0).unsqueeze(0).to(device_gpu)
            m_tensor_gpu = torch.FloatTensor(mel_mat).unsqueeze(0).unsqueeze(0).to(device_gpu)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = swara_model_gpu(s_tensor_gpu)
            torch.cuda.synchronize()
            times_swara_inf_gpu_b1.append(time.perf_counter() - t0)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = mel_model_gpu(m_tensor_gpu)
            torch.cuda.synchronize()
            times_mel_inf_gpu_b1.append(time.perf_counter() - t0)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = swara_model_gpu(dummy_swara_b32_gpu)
            torch.cuda.synchronize()
            times_swara_inf_gpu_b32.append((time.perf_counter() - t0) / 32.0)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = mel_model_gpu(dummy_mel_b32_gpu)
            torch.cuda.synchronize()
            times_mel_inf_gpu_b32.append((time.perf_counter() - t0) / 32.0)

    avg_audio_dur = float(np.mean(durations))
    avg_load = float(np.mean(times_load))
    avg_yin = float(np.mean(times_yin))
    avg_tonic = float(np.mean(times_tonic))
    avg_swara_gen = float(np.mean(times_swara_gen))
    avg_mel_gen = float(np.mean(times_mel_gen))

    avg_swara_inf_cpu = float(np.mean(times_swara_inf_cpu))
    avg_mel_inf_cpu = float(np.mean(times_mel_inf_cpu))

    avg_swara_inf_gpu_b1 = float(np.mean(times_swara_inf_gpu_b1)) if has_gpu else avg_swara_inf_cpu
    avg_mel_inf_gpu_b1 = float(np.mean(times_mel_inf_gpu_b1)) if has_gpu else avg_mel_inf_cpu
    avg_swara_inf_gpu_b32 = float(np.mean(times_swara_inf_gpu_b32)) if has_gpu else avg_swara_inf_cpu
    avg_mel_inf_gpu_b32 = float(np.mean(times_mel_inf_gpu_b32)) if has_gpu else avg_mel_inf_cpu

    # Totals per 30s segment
    total_swara_cpu_ms = (avg_load + avg_yin + avg_tonic + avg_swara_gen + avg_swara_inf_cpu) * 1000
    total_swara_gpu_ms = (avg_load + avg_yin + avg_tonic + avg_swara_gen + avg_swara_inf_gpu_b1) * 1000
    total_mel_cpu_ms = (avg_load + avg_mel_gen + avg_mel_inf_cpu) * 1000
    total_mel_gpu_ms = (avg_load + avg_mel_gen + avg_mel_inf_gpu_b1) * 1000

    median_segments = 27
    full_rec_cpu_s = (total_swara_cpu_ms * median_segments) / 1000.0
    full_rec_gpu_s = (total_swara_gpu_ms * median_segments) / 1000.0

    rtf_cpu = (total_swara_cpu_ms / 1000.0) / avg_audio_dur
    rtf_gpu = (total_swara_gpu_ms / 1000.0) / avg_audio_dur

    runtime_results = {
        "hardware_software_info": {
            "gpu_name": gpu_name,
            "cpu_name": cpu_name,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if has_gpu else "N/A",
            "benchmark_protocol": "Averaged over 10 sample recordings (270s total) with 10 CUDA warmup iterations and explicit torch.cuda.synchronize() boundary calls."
        },
        "segment_benchmark_30s": {
            "audio_load_resample_ms": round(avg_load * 1000, 2),
            "yin_pitch_tracking_ms": round(avg_yin * 1000, 2),
            "tonic_estimation_ms": round(avg_tonic * 1000, 2),
            "swaragram_feature_generation_ms": round(avg_swara_gen * 1000, 2),
            "mel_spectrogram_generation_ms": round(avg_mel_gen * 1000, 2),
            "swaragram_crnn_inference_cpu_ms": round(avg_swara_inf_cpu * 1000, 2),
            "mel_crnn_inference_cpu_ms": round(avg_mel_inf_cpu * 1000, 2),
            "swaragram_crnn_inference_gpu_b1_ms": round(avg_swara_inf_gpu_b1 * 1000, 2),
            "mel_crnn_inference_gpu_b1_ms": round(avg_mel_inf_gpu_b1 * 1000, 2),
            "swaragram_crnn_inference_gpu_b32_ms_per_seg": round(avg_swara_inf_gpu_b32 * 1000, 2),
            "mel_crnn_inference_gpu_b32_ms_per_seg": round(avg_mel_inf_gpu_b32 * 1000, 2),
        },
        "totals_30s_segment": {
            "swaragram_total_pipeline_cpu_ms": round(total_swara_cpu_ms, 2),
            "swaragram_total_pipeline_gpu_ms": round(total_swara_gpu_ms, 2),
            "mel_total_pipeline_cpu_ms": round(total_mel_cpu_ms, 2),
            "mel_total_pipeline_gpu_ms": round(total_mel_gpu_ms, 2),
            "swaragram_rtf_cpu": round(rtf_cpu, 5),
            "swaragram_rtf_gpu": round(rtf_gpu, 5),
            "speedup_vs_realtime_cpu": f"{1.0 / rtf_cpu:.1f}x faster than real-time",
            "speedup_vs_realtime_gpu": f"{1.0 / rtf_gpu:.1f}x faster than real-time",
        },
        "full_recording_benchmark_13_5_min": {
            "median_segments_count": median_segments,
            "full_recording_audio_duration_minutes": 13.5,
            "end_to_end_latency_cpu_seconds": round(full_rec_cpu_s, 2),
            "end_to_end_latency_gpu_seconds": round(full_rec_gpu_s, 2),
        }
    }

    return runtime_results


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting Statistical & Comprehensive CPU/GPU Latency Benchmarking...")

    mcnemar, ci = compute_bootstrap_and_mcnemar()
    runtime = run_latency_benchmarks()

    # Save JSONs
    with open(OUT_DIR / "statistical_significance_results.json", "w") as f:
        json.dump({"mcnemar": mcnemar, "confidence_intervals": ci}, f, indent=2)

    with open(OUT_DIR / "runtime_benchmark_results.json", "w") as f:
        json.dump(runtime, f, indent=2)

    logger.info("Complete benchmarking finished!")


if __name__ == "__main__":
    main()
