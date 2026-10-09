import torch 
import torch.utils.data as data
import pandas as pd 
import numpy as np 
from pathlib import Path

import json

class MelDataset(data.Dataset):
    def __init__(self, split, feature_type="mel",
                 features_root="data/processed/features", 
                 label_mapping_path="data/processed/metadata/label_mapping.json", 
                 stats_path="data/processed/metadata/dataset_stats.json",
                 transform=None):
        self.split = split
        self.feature_type = feature_type
        self.features_root = Path(features_root)
        self.transform = transform

        # Load global label mapping
        with open(label_mapping_path, "r") as f:
            self.raag_to_idx = json.load(f)
            
        self.raags = [raag for raag, idx in sorted(self.raag_to_idx.items(), key=lambda x: x[1])]
        self.num_classes = len(self.raags)
        
        # Only load z-score stats for Mel Spectrograms. 
        # Swaragrams are already normalized in [0, 1] by the Gaussian activation.
        if self.feature_type == "mel":
            with open(stats_path, "r") as f:
                stats = json.load(f)
                self.train_mean = stats["train_mean"]
                self.train_std = stats["train_std"]
        else:
            self.train_mean = None
            self.train_std = None

        # Load the unified segments catalog for the requested feature type
        metadata_path = self.features_root / f"{feature_type}_segments.csv"
        full_df = pd.read_csv(metadata_path)

        # Filter to the specific split (e.g. searching for 'feature_type/train' in the path)
        self.df = full_df[full_df["feature_file"].str.contains(f"{feature_type}[\\\\/]{split}", regex=True)].reset_index(drop=True)

        # Check for missing files
        missing = []
        for _, row in self.df.iterrows():
            feature_rel = Path(*row["feature_file"].replace("\\", "/").split("/"))
            feature_path = self.features_root / feature_rel
            
            if not feature_path.exists():
                missing.append(row["segment_id"])
                
        if missing:
            print(f"Warning: {len(missing)} feature files missing in {split} split")
            print(f"   First few: {missing[:5]}")
            self.df = self.df[~self.df['segment_id'].isin(missing)].reset_index(drop=True)
        
        print(f" {split}: {len(self.df)} samples, {self.num_classes} classes")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Resolve platform-independent feature path
        feature_rel = Path(*row["feature_file"].replace("\\", "/").split("/"))
        feat_path = self.features_root / feature_rel
        
        # Load .npy feature
        spectrogram = np.load(feat_path)
        
        # Dataset-level Normalization (Z-score computed from training set)
        if self.feature_type == "mel" and self.train_mean is not None:
            spectrogram = (spectrogram - self.train_mean) / (self.train_std + 1e-8)
        
        # Convert to tensor, add channel dim: (1, 128, 1292) or (1, 12, T)
        spectrogram = torch.FloatTensor(spectrogram).unsqueeze(0)
        
        # Label
        label_idx = self.raag_to_idx[row['label']]
        label = torch.tensor(label_idx, dtype=torch.long)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
            
        return spectrogram, label


# Quick test
if __name__ == "__main__":
    for split in ['train', 'validation', 'test']:
        try:
            dataset = MelDataset(split=split)
            if len(dataset) > 0:
                spec, label = dataset[0]
                print(f"\n{split}:")
                print(f"  Spectrogram shape: {spec.shape}")
                print(f"  Label: {label} ({dataset.raags[label]})")
                print(f"  Classes: {dataset.num_classes}")
        except Exception as e:
            print(f"\n{split}: ERROR - {e}")
