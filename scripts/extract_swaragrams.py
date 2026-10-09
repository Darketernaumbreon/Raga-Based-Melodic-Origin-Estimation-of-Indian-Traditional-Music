"""
Extract Swaragram Features offline for all dataset splits.
"""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path when executing script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.split_utils import load_dataset_splits
from src.preprocessing.feature_extraction import FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Extract Swaragram Features Offline")
    parser.add_argument("--dataset", type=str, default="A", choices=["A", "B"], help="Dataset to process: 'A' or 'B'")
    parser.add_argument("--out-dir", type=str, default="data/processed/features", help="Output features root directory")
    args = parser.parse_args()

    logger.info(f"=== Starting Offline Swaragram Extraction for Dataset {args.dataset} ===")

    # 1. Load splits (generating recording-level splits if needed)
    train_df, val_df, test_df = load_dataset_splits(dataset_name=args.dataset)
    logger.info(f"Loaded splits | Train: {len(train_df)} recs, Val: {len(val_df)} recs, Test: {len(test_df)} recs")

    # Combine splits for batch extraction
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Combined trainable pool size: {len(df)} recordings")

    # 2. Configure Feature Extractor
    extractor = FeatureExtractor(
        feature_type="swaragram",
        sr=22050,
        hop_length=512,
        num_bins=12,
        sigma=25.0
    )

    # 3. Execute batch offline extraction
    segment_df = extractor.save_batch(
        df=df,
        output_dir=args.out_dir,
        clip_duration=30.0,
        hop_duration=15.0,
        min_duration=10.0
    )

    logger.info(f"=== Extraction Complete! ===")
    logger.info(f"Total segments cached: {len(segment_df)}")
    
    # 4. Quick Verification of a few cached features
    logger.info(f"=== Verifying Cached Features ===")
    if len(segment_df) > 0:
        sample_recs = segment_df.sample(min(5, len(segment_df)), random_state=42)
        for _, s_row in sample_recs.iterrows():
            npy_path = Path(args.out_dir) / s_row["feature_file"]
            if npy_path.exists():
                feat = np.load(npy_path)
                has_nans = np.isnan(feat).any()
                has_infs = np.isinf(feat).any()
                min_val, max_val = feat.min(), feat.max()
                logger.info(
                    f"Segment: {s_row['segment_id']} | "
                    f"Shape: {feat.shape} | "
                    f"NaNs/Infs: {has_nans}/{has_infs} | "
                    f"Range: [{min_val:.4f}, {max_val:.4f}]"
                )
            else:
                logger.warning(f"Feature file missing for verify: {npy_path}")

if __name__ == "__main__":
    main()
