"""
Stratified Dataset Splitting Utilities.

Provides functions to generate and load train, validation, and test splits
stratified at the recording level to ensure no data leakage.
"""

import logging
from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def generate_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Assign train/val/test split to each row in df, stratifying by canonical_raag.
    Modifies the 'split' column of the df and returns it.
    """
    df = df.copy()
    
    # Initialize the split column
    df["split"] = None
    
    # We only split rows where canonical_raag is not null and status is KEEP or OPTIONAL
    # (or just all rows in the dataset metadata file, since they are already filtered)
    valid_mask = df["canonical_raag"].notna()
    valid_df = df[valid_mask].copy()
    
    # First split: train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        valid_df.index,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=valid_df["canonical_raag"]
    )
    
    # Second split: val vs test from temp
    val_relative_ratio = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_relative_ratio,
        random_state=random_state,
        stratify=valid_df.loc[temp_idx, "canonical_raag"]
    )
    
    # Assign split labels
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    
    return df

def load_dataset_splits(
    dataset_name: str = "A",
    metadata_dir: Union[str, Path] = "data/processed/metadata"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test splits for the specified dataset.
    If splits are not already defined in the metadata CSV, generates them and saves back to disk.
    """
    base_dir = Path(metadata_dir)
    ds_dir = base_dir / f"dataset_{dataset_name}"
    csv_path = ds_dir / f"metadata_{dataset_name}.csv"
    
    if not csv_path.exists():
        # Fallback: check if we are running from a subdirectory
        # (e.g. if the working directory is different)
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / base_dir / f"dataset_{dataset_name}" / f"metadata_{dataset_name}.csv"
        
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Check if we need to generate splits
    # If the split column is completely missing, all NaN, or has empty strings
    needs_split = (
        "split" not in df.columns or 
        df["split"].isnull().all() or 
        (df["split"] == "").all() or
        df["split"].astype(str).str.strip().eq("nan").all()
    )
    
    if needs_split:
        logger.info(f"Generating stratified splits for Dataset {dataset_name}...")
        df = generate_stratified_splits(df)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved generated splits back to {csv_path}")
    
    # Filter by split column
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    return train_df, val_df, test_df
