"""
hybrid_dataset.py

PyTorch Dataset for the Dual-Branch (Hybrid) CRNN experiment.
Loads both a Mel Spectrogram and a Swaragram for every segment,
joining the two metadata CSVs on segment_id.
"""
import json
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from pathlib import Path


class HybridDataset(data.Dataset):
    """
    Returns (mel_tensor, swara_tensor), label for every segment.

    Mel features are z-score normalized using training set statistics.
    Swaragram features are already bounded in [0, 1] and require no normalization.
    """
    def __init__(
        self,
        split,
        features_root="data/processed/features",
        label_mapping_path="data/processed/metadata/label_mapping.json",
        stats_path="data/processed/metadata/dataset_stats.json",
    ):
        self.split = split
        self.features_root = Path(features_root)

        # Label mapping
        with open(label_mapping_path, "r") as f:
            self.raag_to_idx = json.load(f)
        self.raags = [r for r, _ in sorted(self.raag_to_idx.items(), key=lambda x: x[1])]
        self.num_classes = len(self.raags)

        # Z-score stats for Mel normalization
        with open(stats_path, "r") as f:
            stats = json.load(f)
            self.train_mean = stats["train_mean"]
            self.train_std = stats["train_std"]

        # Load and filter mel metadata
        mel_df = pd.read_csv(self.features_root / "mel_segments.csv")
        mel_df = mel_df[mel_df["feature_file"].str.contains(
            f"mel[\\\\/]{split}", regex=True)].reset_index(drop=True)

        # Load and filter swaragram metadata
        swara_df = pd.read_csv(self.features_root / "swaragram_segments.csv")
        swara_df = swara_df[swara_df["feature_file"].str.contains(
            f"swaragram[\\\\/]{split}", regex=True)].reset_index(drop=True)

        # Inner join on segment_id to guarantee alignment
        swara_df = swara_df.rename(columns={"feature_file": "swara_file", "frames": "swara_frames"})
        self.df = mel_df.merge(
            swara_df[["segment_id", "swara_file"]],
            on="segment_id",
            how="inner"
        ).reset_index(drop=True)

        print(f"  {split}: {len(self.df)} aligned segments, {self.num_classes} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Mel Spectrogram
        mel_rel = Path(*row["feature_file"].replace("\\", "/").split("/"))
        mel = np.load(self.features_root / mel_rel).astype(np.float32)
        mel = (mel - self.train_mean) / (self.train_std + 1e-8)
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0)  # (1, 128, T)

        # Load Swaragram
        swara_rel = Path(*row["swara_file"].replace("\\", "/").split("/"))
        swara = np.load(self.features_root / swara_rel).astype(np.float32)
        swara_tensor = torch.FloatTensor(swara).unsqueeze(0)  # (1, 12, T)

        # Align time dimension (pad/truncate the shorter one)
        T = min(mel_tensor.shape[-1], swara_tensor.shape[-1])
        mel_tensor = mel_tensor[..., :T]
        swara_tensor = swara_tensor[..., :T]

        # Label
        label_idx = self.raag_to_idx[row["label"]]
        label = torch.tensor(label_idx, dtype=torch.long)

        return (mel_tensor, swara_tensor), label
