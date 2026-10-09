"""
Feature Extraction Utilities for Hindustani Raag Classification.

Provides standard audio feature extraction functions (Mel Spectrogram, CQT, MFCC, Chroma)
and a unified FeatureExtractor class capable of on-the-fly and batch offline extraction.
"""

from .audio_utils import resolve_audio_path
import logging
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Extract log-scaled Mel spectrogram from an audio waveform.

    Returns
    -------
    np.ndarray
        Log-Mel spectrogram array of shape (n_mels, time_steps).
    """
    import librosa
    if fmax is None:
        fmax = float(sr / 2.0)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.astype(np.float32)

def extract_mfcc(
    audio: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract MFCC features from an audio waveform.

    Returns
    -------
    np.ndarray
        MFCC array of shape (n_mfcc, time_steps).
    """
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.astype(np.float32)

def extract_chroma(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_chroma: int = 12
) -> np.ndarray:
    """
    Extract pitch-class profile (Chroma) features, useful for tonic/swara patterns.

    Returns
    -------
    np.ndarray
        Chroma array of shape (n_chroma, time_steps).
    """
    import librosa
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    return chroma.astype(np.float32)

def extract_cqt(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12
) -> np.ndarray:
    """
    Extract Constant-Q Transform (CQT) magnitude spectrogram.
    Widely used in music information retrieval and raag analysis for logarithmic frequency resolution.

    Returns
    -------
    np.ndarray
        Log-CQT magnitude array of shape (n_bins, time_steps).
    """
    import librosa
    cqt = np.abs(librosa.cqt(y=audio, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave))
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    return log_cqt.astype(np.float32)


def normalize_features(feat: np.ndarray, method: str = "standard") -> np.ndarray:
    """
    Normalize feature array (standard score or min-max scaling).
    """
    if method == "standard":
        mean = feat.mean()
        std = feat.std()
        if std > 1e-8:
            return (feat - mean) / std
        return feat - mean
    elif method == "minmax":
        min_val = feat.min()
        max_val = feat.max()
        if max_val - min_val > 1e-8:
            return (feat - min_val) / (max_val - min_val)
        return feat - min_val
    return feat


def extract_swaragram(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    num_bins: int = 12,
    sigma: float = 25.0
) -> np.ndarray:
    """
    Extracts a 12-bin Pitch-Class Swaragram normalized to the estimated tonic.
    """
    from src.swaragram.pitch_extractor import PitchExtractor
    from src.swaragram.tonic_estimator import TonicEstimator
    from src.swaragram.pitch_normalizer import PitchNormalizer
    from src.swaragram.swaragram_generator import SwaragramGenerator
    
    extractor = PitchExtractor(sr=sr, hop_length=hop_length)
    f0 = extractor.extract(audio)
    
    tonic_estimator = TonicEstimator()
    tonic_hz = tonic_estimator.estimate(f0)
    
    normalizer = PitchNormalizer()
    cents = normalizer.normalize(f0, tonic_hz)
    
    generator = SwaragramGenerator(num_bins=num_bins, sigma=sigma)
    swaragram = generator.generate(cents)
    
    return swaragram

class FeatureExtractor:
    """
    Unified Feature Extractor for single files or batch dataset processing.
    """

    def __init__(
        self,
        feature_type: str = "mel",
        sr: int = 22050,
        hop_length: int = 512,
        feature_normalize: Optional[str] = None,
        clip_duration: Optional[float] = 30.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        feature_type : str
            Type of feature to extract: 'mel', 'mfcc', 'chroma', 'cqt', or 'swaragram'.
        sr : int
            Sampling rate in Hz.
        hop_length : int
            Hop length for STFT/CQT calculations.
        feature_normalize : Optional[str]
            Normalization method for features: 'standard', 'minmax', or None.
        clip_duration : Optional[float]
            Target clip duration in seconds, by default 30.0.
        **kwargs
            Additional arguments passed to specific feature extractors.
        """
        self.feature_type = feature_type.lower()
        self.sr = sr
        self.hop_length = hop_length
        self.feature_normalize = feature_normalize.lower() if feature_normalize is not None else None
        self.clip_duration = clip_duration
        self.kwargs = kwargs

    def transform(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from an audio array."""
        if self.feature_type == "mel":
            feat = extract_mel_spectrogram(audio, sr=self.sr, hop_length=self.hop_length, **self.kwargs)
        elif self.feature_type == "mfcc":
            feat = extract_mfcc(audio, sr=self.sr, hop_length=self.hop_length, **self.kwargs)
        elif self.feature_type == "chroma":
            feat = extract_chroma(audio, sr=self.sr, hop_length=self.hop_length, **self.kwargs)
        elif self.feature_type == "cqt":
            feat = extract_cqt(audio, sr=self.sr, hop_length=self.hop_length, **self.kwargs)
        elif self.feature_type == "swaragram":
            feat = extract_swaragram(audio, sr=self.sr, hop_length=self.hop_length, **self.kwargs)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

        if self.feature_normalize is not None:
            feat = normalize_features(feat, method=self.feature_normalize)

        return feat

    def process_file(
        self,
        file_path: Union[str, Path],
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """Load audio from file and extract features."""
        from .audio_utils import load_and_resample_audio
        audio_path = resolve_audio_path(file_path)
        audio, _ = load_and_resample_audio(
            audio_path,
            target_sr=self.sr,
            offset=offset,
            duration=duration,
            normalize=True,
            pad_to_duration=duration
        )
        return self.transform(audio)

    def process_recording(
        self,
        file_path: Union[str, Path],
        clip_duration: float = 30.0,
        hop_duration: float = 15.0,
        min_duration: float = 10.0
    ) -> List[Tuple[np.ndarray, float, float, int]]:
        """
        Process a recording: load, normalize, segment, and extract features per segment.
        Returns a list of tuples: (feature_array, offset, duration, segment_idx)
        """
        from .audio_utils import load_and_resample_audio, resolve_audio_path, segment_waveform_with_metadata
        
        audio_path = resolve_audio_path(file_path)
        audio, sr = load_and_resample_audio(
            audio_path,
            target_sr=self.sr,
            normalize=True
        )
        
        segments = segment_waveform_with_metadata(
            audio,
            sr=self.sr,
            segment_duration_s=clip_duration,
            hop_duration_s=hop_duration,
            min_duration_s=min_duration
        )
        
        results = []
        for idx, (seg_audio, offset, duration) in enumerate(segments):
            feat = self.transform(seg_audio)
            results.append((feat, offset, duration, idx))
            
        return results

    def save_batch(
        self,
        df: "pd.DataFrame",
        output_dir: Union[str, Path],
        id_column: str = "recording_id",
        path_column: str = "audio_path",
        label_column: str = "canonical_raag",
        clip_duration: float = 30.0,
        hop_duration: float = 15.0,
        min_duration: float = 10.0
    ) -> "pd.DataFrame":
        """
        Extract features for all recordings in a DataFrame, save segments as .npy files,
        and save a lightweight metadata CSV alongside stats and extraction logs.
        Supports resume by skipping recordings that are already fully processed.
        """
        from tqdm import tqdm
        out_path = Path(output_dir) / self.feature_type

        # Load existing segment metadata CSV for fast resume check
        metadata_csv_path = Path(output_dir) / f"{self.feature_type}_segments.csv"
        existing_records = {}
        if metadata_csv_path.exists():
            try:
                old_df = pd.read_csv(metadata_csv_path)
                for _, row_rec in old_df.iterrows():
                    rec_id_str = str(row_rec["recording_id"])
                    if rec_id_str not in existing_records:
                        existing_records[rec_id_str] = []
                    existing_records[rec_id_str].append(row_rec.to_dict())
            except Exception as e:
                logger.warning(f"Could not load existing metadata CSV for resume check: {e}")

        segment_records = []
        stats = {
            "total_recordings": len(df),
            "succeeded": 0,
            "failed": 0,
            "total_segments": 0,
            "total_duration_s": 0.0,
            "errors": []
        }

        logger.info(f"Extracting '{self.feature_type}' features for {len(df)} recordings -> {out_path}")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {self.feature_type}"):
            rec_id = str(row[id_column])
            audio_path = row[path_column]
            label = row[label_column] if label_column in row else None
            
            # Determine split subdirectory structure
            split_subdir = ""
            if "split" in row and pd.notna(row["split"]):
                split_val = str(row["split"]).strip().lower()
                if split_val == "val":
                    split_subdir = "validation"
                else:
                    split_subdir = split_val

            # Resume check: if recording is in existing records and all segment files exist, skip extraction
            if rec_id in existing_records:
                all_exist = True
                for rec_item in existing_records[rec_id]:
                    f_file = Path(output_dir) / rec_item["feature_file"]
                    if not f_file.exists():
                        all_exist = False
                        break
                if all_exist:
                    segment_records.extend(existing_records[rec_id])
                    stats["total_segments"] += len(existing_records[rec_id])
                    stats["total_duration_s"] += sum(item["duration"] for item in existing_records[rec_id])
                    stats["succeeded"] += 1
                    continue

            try:
                segments = self.process_recording(
                    audio_path,
                    clip_duration=clip_duration,
                    hop_duration=hop_duration,
                    min_duration=min_duration
                )
                
                for feat, offset, duration, seg_idx in segments:
                    seg_id = f"{rec_id}_seg{seg_idx:03d}"
                    
                    split_out_path = out_path / split_subdir
                    split_out_path.mkdir(parents=True, exist_ok=True)
                    save_file = split_out_path / f"{seg_id}.npy"
                    
                    np.save(save_file, feat)
                    
                    segment_records.append({
                        "recording_id": rec_id,
                        "segment_id": seg_id,
                        "label": label,
                        "feature_file": str(Path(self.feature_type) / split_subdir / f"{seg_id}.npy"),
                        "offset": offset,
                        "duration": duration,
                        "frames": feat.shape[1] if feat.ndim > 1 else feat.shape[0]
                    })
                    stats["total_segments"] += 1
                    stats["total_duration_s"] += duration
                
                stats["succeeded"] += 1
            except Exception as e:
                stats["failed"] += 1
                error_msg = f"Error extracting features for {rec_id} ({audio_path}): {e}"
                logger.warning(error_msg)
                stats["errors"].append(error_msg)

        # Save segment metadata CSV
        segment_df = pd.DataFrame(segment_records)
        metadata_csv_path = Path(output_dir) / f"{self.feature_type}_segments.csv"
        segment_df.to_csv(metadata_csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved segment metadata CSV to {metadata_csv_path}")
        
        # Save feature extraction log / stats
        stats_path = Path(output_dir) / f"{self.feature_type}_extraction_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            import json
            json.dump(stats, f, indent=4)
        logger.info(f"Saved extraction stats to {stats_path}")
        
        # Save errors to a log file if any occurred
        if stats["errors"]:
            log_file_path = Path("logs") / "feature_extraction.log"
            log_file_path.parent.mkdir(exist_ok=True)
            with open(log_file_path, "a", encoding="utf-8") as lf:
                from datetime import datetime
                lf.write(f"\n--- Feature Extraction Run at {datetime.now().isoformat()} ---\n")
                for err in stats["errors"]:
                    lf.write(err + "\n")
            logger.info(f"Appended errors to feature extraction log: {log_file_path}")
            
        return segment_df
