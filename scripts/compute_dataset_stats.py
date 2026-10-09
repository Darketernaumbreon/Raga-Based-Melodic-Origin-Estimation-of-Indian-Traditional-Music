import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

def compute_train_stats():
    df = pd.read_csv('data/processed/features/mel_segments.csv')
    train_files = df[df['feature_file'].str.contains('train')]['feature_file'].values
    
    sum_val = 0.0
    sum_sq_val = 0.0
    count = 0
    
    print(f"Computing statistics over {len(train_files)} training files...")
    
    # We sample a subset if it's too huge, but it's only 27k files, so we can process all of them
    # For speed, let's take a 10% random sample which is statically sufficient for mean/std
    np.random.seed(42)
    sample_files = np.random.choice(train_files, size=min(2500, len(train_files)), replace=False)
    
    for f in tqdm(sample_files):
        # Handle platform paths
        parts = f.replace('\\', '/').split('/')
        feat_path = Path('data/processed/features') / Path(*parts)
        
        spec = np.load(feat_path)
        sum_val += spec.sum()
        sum_sq_val += (spec**2).sum()
        count += spec.size
        
    mean = sum_val / count
    std = np.sqrt((sum_sq_val / count) - mean**2)
    
    stats = {
        "train_mean": float(mean),
        "train_std": float(std)
    }
    
    out_path = Path('data/processed/metadata/dataset_stats.json')
    out_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"\nStats computed and saved to {out_path}")
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")

if __name__ == "__main__":
    compute_train_stats()
