"""
Extract Swaragram Features offline for all dataset splits using Multiprocessing.
"""
import argparse
import logging
import sys
import os

# MUST be set before numpy is imported to prevent thread thrashing in multiprocessing!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm

# Ensure project root is in sys.path when executing script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.split_utils import load_dataset_splits
from src.preprocessing.feature_extraction import FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def process_single_row(row_tuple, extractor_kwargs, out_dir, split_subdir, existing_records):
    """
    Worker function to process a single recording.
    """
    idx, row = row_tuple
    rec_id = str(row["recording_id"])
    audio_path = row["audio_path"]
    label = row.get("canonical_raag", None)
    
    # Resume check
    if rec_id in existing_records:
        all_exist = True
        for rec_item in existing_records[rec_id]:
            f_file = Path(out_dir) / rec_item["feature_file"]
            if not f_file.exists():
                all_exist = False
                break
        if all_exist:
            return "skipped", existing_records[rec_id]

    # Initialize a new extractor for this worker (safe for multiprocessing)
    extractor = FeatureExtractor(**extractor_kwargs)
    
    try:
        segments = extractor.process_recording(
            audio_path,
            clip_duration=30.0,
            hop_duration=15.0,
            min_duration=10.0
        )
        
        records = []
        out_path = Path(out_dir) / extractor.feature_type / split_subdir
        out_path.mkdir(parents=True, exist_ok=True)
        
        for feat, offset, duration, seg_idx in segments:
            seg_id = f"{rec_id}_seg{seg_idx:03d}"
            save_file = out_path / f"{seg_id}.npy"
            np.save(save_file, feat)
            
            records.append({
                "recording_id": rec_id,
                "segment_id": seg_id,
                "label": label,
                "feature_file": str(Path(extractor.feature_type) / split_subdir / f"{seg_id}.npy"),
                "offset": offset,
                "duration": duration,
                "frames": feat.shape[1] if feat.ndim > 1 else feat.shape[0]
            })
        return "success", records
    except Exception as e:
        return "error", f"Error extracting {rec_id} ({audio_path}): {e}"


def main():
    parser = argparse.ArgumentParser(description="Extract Swaragram Features Offline (Parallel)")
    parser.add_argument("--dataset", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--out-dir", type=str, default="data/processed/features")
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 1)
    parser.add_argument("--force", action="store_true", help="Force re-extraction, ignoring resume cache")
    args = parser.parse_args()

    logger.info(f"=== Starting Parallel Swaragram Extraction ({args.workers} workers) ===")

    train_df, val_df, test_df = load_dataset_splits(dataset_name=args.dataset)
    train_df["split_subdir"] = "train"
    val_df["split_subdir"] = "validation"
    test_df["split_subdir"] = "test"
    
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Combined trainable pool size: {len(df)} recordings")

    metadata_csv_path = Path(args.out_dir) / "swaragram_segments.csv"
    existing_records = {}
    if not args.force and metadata_csv_path.exists():
        try:
            old_df = pd.read_csv(metadata_csv_path)
            for _, row_rec in old_df.iterrows():
                rec_id_str = str(row_rec["recording_id"])
                if rec_id_str not in existing_records:
                    existing_records[rec_id_str] = []
                existing_records[rec_id_str].append(row_rec.to_dict())
            logger.info(f"Resume mode: found {len(existing_records)} existing records.")
        except Exception as e:
            logger.warning(f"Could not load metadata CSV for resume check: {e}")
    elif args.force:
        logger.info("--force flag set: ignoring resume cache, re-extracting all recordings.")

    extractor_kwargs = {
        "feature_type": "swaragram",
        "sr": 22050,
        "hop_length": 512,
        "num_bins": 12,
        "sigma": 25.0
    }

    tasks = []
    for idx, row in df.iterrows():
        tasks.append(((idx, row), extractor_kwargs, args.out_dir, row["split_subdir"], existing_records))

    segment_records = []
    success_count = 0
    skip_count = 0
    error_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_row, *task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Extracting"):
            status, result = future.result()
            if status == "success":
                segment_records.extend(result)
                success_count += 1
            elif status == "skipped":
                segment_records.extend(result)
                skip_count += 1
            elif status == "error":
                logger.error(result)
                error_count += 1

    if segment_records:
        pd.DataFrame(segment_records).to_csv(metadata_csv_path, index=False)
        logger.info(f"Saved metadata to {metadata_csv_path}")

    logger.info(f"=== Extraction Complete! ===")
    logger.info(f"Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")
    logger.info(f"Total segments cached: {len(segment_records)}")

if __name__ == "__main__":
    main()
