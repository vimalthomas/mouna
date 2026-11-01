#!/usr/bin/env python3
"""Extend keypoints_official_split_v2norm.parquet with ASL Citizen data.

Starts from the already-normalized v2norm dataset (WLASL MediaPipe + WLASL
OpenPose-recovered + official MS-ASL) and appends per-part-normalized ASL
Citizen rows. All ASL Citizen rows are tagged official_split="train" (same
treatment as MS-ASL) so the fixed WLASL val/test split used for every
comparison so far stays untouched -- this only grows the training set.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from mouna.data.part_normalize import normalize_per_part  # noqa: E402

BASE = Path("data/silver/keypoints/keypoints_official_split_v2norm.parquet")
RAW_ASLC = Path("data/silver/keypoints/keypoints_asl_citizen_raw.parquet")
OUTPUT = Path("data/silver/keypoints/keypoints_official_split_v2norm_aslc.parquet")


def main():
    base_df = pd.read_parquet(BASE)
    print(f"Base (v2norm): {len(base_df)} sequences")

    raw_aslc = pd.read_parquet(RAW_ASLC)
    raw_aslc = raw_aslc[raw_aslc["success"]]

    rows = []
    for _, row in raw_aslc.iterrows():
        flat = pickle.loads(row["keypoints_pickle"])
        normed = normalize_per_part(flat)
        rows.append(
            {
                "video_id": row["video_id"],
                "gloss": row["gloss"],
                "keypoints_pickle": pickle.dumps(normed),
                "num_frames": len(normed),
                "success": True,
                "official_split": "train",
                "source": "asl_citizen",
            }
        )
    aslc_df = pd.DataFrame(rows)
    print(f"ASL Citizen: {len(aslc_df)} sequences, {aslc_df['gloss'].nunique()} classes")

    df = pd.concat([base_df, aslc_df], ignore_index=True)
    df.to_parquet(OUTPUT, index=False)

    print(f"\nTotal: {len(df)} sequences, {df['gloss'].nunique()} classes -> {OUTPUT}")
    print(df["official_split"].value_counts())
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()
