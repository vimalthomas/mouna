#!/usr/bin/env python3
"""Re-test palm orientation + velocity on top of the now-validated
handedness-canonicalized dataset (v4canon). The orientation feature's
earlier null result (v3motion vs v2norm, not significant) was almost
certainly confounded by handedness mirroring -- we proved orientation
flips to a different vector slot with partially-flipped sign depending on
which hand is dominant in the raw footage. Canonicalization removes that
confound, so this is orientation's fair rematch.
"""

import pickle
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, "src")
from mouna.data.motion_features import add_motion_features  # noqa: E402

INPUT = Path("data/silver/keypoints/keypoints_official_split_v4canon.parquet")
OUTPUT = Path("data/silver/keypoints/keypoints_official_split_v5canonmotion.parquet")


def main():
    df = pd.read_parquet(INPUT)
    rows = []
    for _, row in df.iterrows():
        flat = pickle.loads(row["keypoints_pickle"])
        augmented = add_motion_features(flat)
        rows.append({**row.to_dict(), "keypoints_pickle": pickle.dumps(augmented)})
    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUT, index=False)
    print(f"{len(out)} sequences -> {OUTPUT}")
    sample = pickle.loads(out.iloc[0]["keypoints_pickle"])
    print(f"feature dim per frame: {sample.shape[1]} (expect 456)")


if __name__ == "__main__":
    main()
