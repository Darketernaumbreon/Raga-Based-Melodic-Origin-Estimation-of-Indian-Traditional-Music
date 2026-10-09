"""
Audio Loading and Standardization Utilities.

Provides robust functions to load audio recordings across formats (WAV, MP3),
standardize sampling rates, convert stereo to mono, and generate audio windows.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Union
import numpy as np

logger = logging.getLogger(__name__)

def load_and_resample_audio(
    file_path: Union[str, Path],
    target_sr: int = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None,
    normalize: bool = True,
    pad_to_duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sampling rate.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the audio file (.wav or .mp3).
    target_sr : int, optional
        Target sampling rate in Hz, by default 22050.
    mono : bool, optional
        If True, convert multi-channel audio to mono, by default True.
    offset : float, optional
        Start reading after this time (in seconds), by default 0.0.
    duration : Optional[float], optional
        Only load up to this duration (in seconds), by default None.
    normalize : bool, optional
        If True, peak-normalizes the loaded waveform, by default True.
    pad_to_duration : Optional[float], optional
        If set, pads or trims the waveform to exactly this duration in seconds, by default None.

    Returns
    -------
    Tuple[np.ndarray, int]
        Audio time-series array (float32) and the sampling rate.
    """
    import librosa
    path_str = str(file_path)
    try:
        audio, sr = librosa.load(
            path_str,
            sr=target_sr,
            mono=mono,
            offset=offset,
            duration=duration
        )
        audio = audio.astype(np.float32)
        
        if normalize:
            audio = normalize_waveform(audio)
            
        if pad_to_duration is not None:
            target_length = int(pad_to_duration * target_sr)
            audio = pad_trim_waveform(audio, target_length)
            
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio file {path_str}: {e}")
        raise

def segment_audio_waveform(
    audio: np.ndarray,
    sr: int = 22050,
    segment_duration_s: float = 30.0,
    hop_duration_s: float = 15.0,
    min_duration_s: float = 10.0
) -> List[np.ndarray]:
    """
    Split a long audio waveform into fixed-duration segments with overlap.

    Parameters
    ----------
    audio : np.ndarray
        Input 1D audio time-series array.
    sr : int, optional
        Sampling rate in Hz, by default 22050.
    segment_duration_s : float, optional
        Length of each window in seconds, by default 30.0.
    hop_duration_s : float, optional
        Step size between successive windows in seconds, by default 15.0.
    min_duration_s : float, optional
        Minimum duration required to keep a trailing segment, by default 10.0.

    Returns
    -------
    List[np.ndarray]
        List of audio segment arrays.
    """
    seg_samples = int(segment_duration_s * sr)
    hop_samples = int(hop_duration_s * sr)
    min_samples = int(min_duration_s * sr)

    if len(audio) < min_samples:
        return []
    if len(audio) < seg_samples:
        # Pad with zeros to meet segment duration
        pad_width = seg_samples - len(audio)
        return [np.pad(audio, (0, pad_width), mode='constant')]

    segments = []
    for start in range(0, len(audio) - seg_samples + 1, hop_samples):
        segments.append(audio[start : start + seg_samples])

    # Check remaining trailing audio
    last_end = start + seg_samples
    if len(audio) - last_end >= min_samples:
        trailing = audio[last_end:]
        pad_width = seg_samples - len(trailing)
        segments.append(np.pad(trailing, (0, pad_width), mode='constant'))

    return segments

def resolve_audio_path(relative_path: Union[str, Path]) -> Path:
    """
    Resolve a relative audio path against the project's data/raw directory.
    Handles paths starting with or without 'data/raw'.
    """
    relative = Path(relative_path)
    # Project root is two levels up from this file (src/preprocessing/audio_utils.py)
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "data" / "raw"

    # Remove leading 'data/raw' parts if present to avoid duplication
    if len(relative.parts) >= 2 and relative.parts[0] == "data" and relative.parts[1] == "raw":
        relative = Path(*relative.parts[2:])
    elif len(relative.parts) >= 1 and relative.parts[0] == "raw":
        relative = Path(*relative.parts[1:])

    return raw_data_dir / relative

def normalize_waveform(audio: np.ndarray) -> np.ndarray:
    """
    Peak normalize the audio waveform to range [-1.0, 1.0].
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0.0:
        return audio / max_val
    return audio

def pad_trim_waveform(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or trim waveform to target_length samples.
    """
    if len(audio) < target_length:
        pad_width = target_length - len(audio)
        return np.pad(audio, (0, pad_width), mode='constant')
    elif len(audio) > target_length:
        return audio[:target_length]
    return audio

def segment_waveform_with_metadata(
    audio: np.ndarray,
    sr: int = 22050,
    segment_duration_s: float = 30.0,
    hop_duration_s: float = 15.0,
    min_duration_s: float = 10.0
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Segment audio waveform and return tuples of (segment_audio, offset_seconds, duration_seconds).
    Pads trailing segments to segment_duration_s if they are longer than min_duration_s.
    """
    seg_samples = int(segment_duration_s * sr)
    hop_samples = int(hop_duration_s * sr)
    min_samples = int(min_duration_s * sr)
    
    results = []
    if len(audio) < min_samples:
        return results
        
    if len(audio) < seg_samples:
        padded = pad_trim_waveform(audio, seg_samples)
        results.append((padded, 0.0, len(audio) / sr))
        return results
        
    start = 0
    while start + seg_samples <= len(audio):
        segment = audio[start : start + seg_samples]
        offset = start / sr
        duration = segment_duration_s
        results.append((segment, offset, duration))
        start += hop_samples
        
    # Check remaining trailing audio
    last_end = start
    if len(audio) - last_end >= min_samples:
        trailing = audio[last_end:]
        padded = pad_trim_waveform(trailing, seg_samples)
        offset = last_end / sr
        duration = len(trailing) / sr
        results.append((padded, offset, duration))
        
    return results

