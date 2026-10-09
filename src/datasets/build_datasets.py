"""
Dataset Builder Utility.

Builds Dataset A (>=15 recordings threshold) and Dataset B (>=10 recordings threshold)
from the finalized master metadata file (data/processed/metadata/unified/final_metadata.csv).
Generates class summaries and comparison tables.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def build_training_datasets(
    final_path: Union[str, Path] = "data/processed/metadata/unified/final_metadata.csv",
    out_dir_a: Union[str, Path] = "data/processed/metadata/dataset_A",
    out_dir_b: Union[str, Path] = "data/processed/metadata/dataset_B",
    report_dir: Union[str, Path] = "data/processed/reports",
    threshold_a: int = 15,
    threshold_b: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Dataset A and Dataset B CSVs from final_metadata.csv.
    """
    final_p = Path(final_path)
    dir_a = Path(out_dir_a)
    dir_b = Path(out_dir_b)
    rep_dir = Path(report_dir)

    for d in [dir_a, dir_b, rep_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(final_p, encoding="utf-8")
    logger.info(f"Loaded {len(df)} records from {final_p}")

    trainable = df[df["status"].isin(["KEEP", "OPTIONAL"])].copy()
    logger.info(f"Trainable pool (KEEP+OPTIONAL): {len(trainable)} recordings")

    class_stats = (
        trainable.groupby("canonical_raag")
        .agg(
            recordings=("recording_id", "count"),
            duration_h=("duration_seconds", lambda x: round(x.sum() / 3600, 3)),
            avg_duration_s=("duration_seconds", lambda x: round(x.mean(), 1)),
            datasets=("dataset", lambda x: "; ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values("recordings", ascending=False)
        .reset_index(drop=True)
    )

    # Dataset A
    classes_a = set(class_stats[class_stats["recordings"] >= threshold_a]["canonical_raag"])
    meta_a = trainable[trainable["canonical_raag"].isin(classes_a)].copy()
    summary_a = class_stats[class_stats["canonical_raag"].isin(classes_a)].copy()
    summary_a["dataset_label"] = "A"

    meta_a.to_csv(dir_a / "metadata_A.csv", index=False, encoding="utf-8")
    summary_a.to_csv(dir_a / "class_summary_A.csv", index=False, encoding="utf-8")
    logger.info(f"Saved Dataset A (>={threshold_a} recs): {len(classes_a)} classes, {len(meta_a)} recordings -> {dir_a / 'metadata_A.csv'}")

    # Dataset B
    classes_b = set(class_stats[class_stats["recordings"] >= threshold_b]["canonical_raag"])
    meta_b = trainable[trainable["canonical_raag"].isin(classes_b)].copy()
    summary_b = class_stats[class_stats["canonical_raag"].isin(classes_b)].copy()
    summary_b["dataset_label"] = "B"

    meta_b.to_csv(dir_b / "metadata_B.csv", index=False, encoding="utf-8")
    summary_b.to_csv(dir_b / "class_summary_B.csv", index=False, encoding="utf-8")
    logger.info(f"Saved Dataset B (>={threshold_b} recs): {len(classes_b)} classes, {len(meta_b)} recordings -> {dir_b / 'metadata_B.csv'}")

    # Comparison Table
    comp = class_stats.copy()
    comp["in_A"] = comp["canonical_raag"].isin(classes_a)
    comp["in_B"] = comp["canonical_raag"].isin(classes_b)
    comp["present_in"] = comp.apply(
        lambda r: "A + B" if r["in_A"] else ("B only" if r["in_B"] else "neither"), axis=1
    )
    comp.to_csv(rep_dir / "dataset_comparison.csv", index=False, encoding="utf-8")
    logger.info(f"Saved dataset_comparison.csv -> {rep_dir / 'dataset_comparison.csv'}")

    return meta_a, meta_b

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_training_datasets()
