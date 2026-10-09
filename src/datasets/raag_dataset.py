"""
PyTorch Dataset and DataLoader implementations for Raag Classification.

Supports loading pre-extracted feature arrays (.npy) or on-the-fly feature extraction
from raw audio (.wav/.mp3) across Dataset A and Dataset B.
"""

import logging
from pathlib import Path
from typing import Dict, Union, Optional, Tuple, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..preprocessing.feature_extraction import FeatureExtractor
from ..preprocessing.audio_utils import load_and_resample_audio

logger = logging.getLogger(__name__)

class RaagDataset(Dataset):
    """
    PyTorch Dataset for Hindustani Raag Classification.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_dir: Optional[Union[str, Path]] = None,
        feature_type: str = "mel",
        sr: int = 22050,
        label_to_idx: Optional[Dict[str, int]] = None,
        extract_on_the_fly: bool = False,
        clip_duration: float = 30.0,
        hop_duration: float = 15.0,
        min_duration: float = 10.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Metadata dataframe containing 'audio_path', 'recording_id', and 'canonical_raag'.
        feature_dir : Optional[Union[str, Path]], optional
            Directory where pre-extracted features (.npy) are stored, by default None.
        feature_type : str, optional
            Type of feature: 'mel', 'mfcc', 'cqt', or 'chroma', by default 'mel'.
        sr : int, optional
            Sampling rate in Hz, by default 22050.
        label_to_idx : Optional[Dict[str, int]], optional
            Mapping from class string to integer ID. If None, builds from unique labels sorted alphabetically.
        extract_on_the_fly : bool, optional
            If True or if pre-extracted .npy file missing, extracts feature directly from audio file, by default False.
        clip_duration : float, optional
            Target segment duration in seconds, by default 30.0.
        hop_duration : float, optional
            Hop size between successive segments in seconds, by default 15.0.
        min_duration : float, optional
            Minimum segment duration in seconds to keep, by default 10.0.
        **kwargs
            Additional arguments passed to FeatureExtractor when extracting on the fly.
        """
        self.df = df.reset_index(drop=True)
        self.feature_dir = Path(feature_dir) if feature_dir is not None else None
        self.feature_type = feature_type.lower()
        self.sr = sr
        self.extract_on_the_fly = extract_on_the_fly
        self.clip_duration = clip_duration
        self.hop_duration = hop_duration
        self.min_duration = min_duration
        
        self.extractor = FeatureExtractor(
            feature_type=self.feature_type,
            sr=self.sr,
            clip_duration=clip_duration,
            **kwargs
        )

        # Build label dictionary sorted alphabetically for canonical reproducibility
        if label_to_idx is None:
            unique_classes = sorted(self.df["canonical_raag"].dropna().unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Build segment index mapping
        self.segments = []
        segment_csv_path = self.feature_dir / f"{self.feature_type}_segments.csv" if self.feature_dir is not None else None
        
        if segment_csv_path is not None and segment_csv_path.exists() and not self.extract_on_the_fly:
            logger.info(f"Loading segment metadata from {segment_csv_path}")
            all_segments_df = pd.read_csv(segment_csv_path)
            
            # Filter segments belonging to recording_ids in the split df
            valid_recording_ids = set(self.df["recording_id"].astype(str))
            filtered_segments = all_segments_df[all_segments_df["recording_id"].astype(str).isin(valid_recording_ids)]
            
            for _, row in filtered_segments.iterrows():
                self.segments.append({
                    "recording_id": str(row["recording_id"]),
                    "segment_id": str(row["segment_id"]),
                    "audio_path": None,  # Not needed since we load pre-extracted npy
                    "label_str": row["label"],
                    "feature_file": row["feature_file"],
                    "offset": row["offset"],
                    "duration": row["duration"]
                })
        else:
            logger.info("Generating segment offsets on-the-fly...")
            for _, row in self.df.iterrows():
                rec_id = str(row["recording_id"])
                audio_path = row["audio_path"]
                label_str = row["canonical_raag"]
                duration = row["duration_seconds"]
                split_val = row["split"] if "split" in row else None
                
                offsets = self._get_segment_offsets(duration)
                for seg_idx, (offset, dur) in enumerate(offsets):
                    self.segments.append({
                        "recording_id": rec_id,
                        "segment_id": f"{rec_id}_seg{seg_idx:03d}",
                        "audio_path": audio_path,
                        "label_str": label_str,
                        "feature_file": None,
                        "offset": offset,
                        "duration": dur,
                        "split": split_val
                    })
                    
        logger.info(f"Initialized RaagDataset with {len(self.df)} recordings, mapped to {len(self.segments)} segments.")

    def _get_segment_offsets(self, duration: float) -> List[Tuple[float, float]]:
        if duration < self.min_duration:
            return []
        if duration < self.clip_duration:
            return [(0.0, duration)]
        
        offsets = []
        start = 0.0
        while start + self.clip_duration <= duration:
            offsets.append((start, self.clip_duration))
            start += self.hop_duration
            
        if duration - start >= self.min_duration:
            offsets.append((start, duration - start))
        return offsets

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seg = self.segments[idx]
        label_idx = self.label_to_idx[seg["label_str"]]
        
        feat_tensor: Optional[torch.Tensor] = None
        
        # Try loading pre-extracted feature array if available
        if self.feature_dir is not None:
            if seg["feature_file"] is not None:
                npy_path = self.feature_dir / seg["feature_file"]
            else:
                # If feature_file is None, construct expected fallback cache path using split subdir
                split_subdir = ""
                if seg.get("split") is not None:
                    split_val = str(seg["split"]).strip().lower()
                    if split_val == "val":
                        split_subdir = "validation"
                    else:
                        split_subdir = split_val
                npy_path = self.feature_dir / self.feature_type / split_subdir / f"{seg['segment_id']}.npy"

            if npy_path.exists():
                try:
                    feat_arr = np.load(npy_path)
                    feat_tensor = torch.from_numpy(feat_arr).float()
                except Exception as e:
                    logger.warning(f"Corrupted cache file {npy_path}: {e}. Extracting on-the-fly.")
                    
        # Extract on-the-fly
        if feat_tensor is None:
            audio_path = seg["audio_path"]
            offset = seg["offset"]
            duration = seg["duration"]
            
            # process_file handles resolving the path, loading, normalizing, resampling, padding, and transforming
            feat_arr = self.extractor.process_file(audio_path, offset=offset, duration=duration)
            feat_tensor = torch.from_numpy(feat_arr).float()
            
        # Add channel dimension (1, n_features, time_steps) if 2D
        if feat_tensor.dim() == 2:
            feat_tensor = feat_tensor.unsqueeze(0)
            
        return feat_tensor, label_idx

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
