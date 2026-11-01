#!/usr/bin/env python3
"""Automated QA layer 2: sign-plausibility flags from the merged, normalized
dataset. Catches a different failure mode than qa_flag_raw.py -- a clip can
have perfect hand detection and still be the wrong sign.

Reuses the hand-feature between/within-class distance method from the
mother/brown vs accident/play confusion analysis: summarize each sequence by
its mean-pooled hand features, then flag a clip whose features sit closer to
a DIFFERENT word's centroid than to its own, or are a statistical outlier
even within its own class.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

# hand dims are the last 126 of the 225-dim per-frame vector (pose=99, then
# left_hand=63, right_hand=63) -- sign identity concentrates here, not pose.
HAND_START = 99
MIN_CLASS_SIZE_FOR_OUTLIER_STAT = 5


def mean_hand_features(flat: np.ndarray) -> np.ndarray:
    return flat[:, HAND_START:].mean(axis=0)


def main(dataset_path: str):
    df = pd.read_parquet(dataset_path)
    df = df[df["success"]].reset_index(drop=True)

    vecs = np.stack([mean_hand_features(pickle.loads(p)) for p in df["keypoints_pickle"]])
    glosses = df["gloss"].values

    centroids = {}
    for g in np.unique(glosses):
        centroids[g] = vecs[glosses == g].mean(axis=0)

    centroid_matrix = np.stack(list(centroids.values()))
    centroid_glosses = list(centroids.keys())

    dist_own = np.zeros(len(df))
    dist_nearest_other = np.zeros(len(df))
    nearest_other_gloss = [None] * len(df)

    for i in range(len(df)):
        g = glosses[i]
        dist_own[i] = np.linalg.norm(vecs[i] - centroids[g])
        dists_to_all = np.linalg.norm(centroid_matrix - vecs[i], axis=1)
        own_idx = centroid_glosses.index(g)
        dists_to_all[own_idx] = np.inf
        nearest_idx = np.argmin(dists_to_all)
        dist_nearest_other[i] = dists_to_all[nearest_idx]
        nearest_other_gloss[i] = centroid_glosses[nearest_idx]

    df["dist_own"] = dist_own
    df["dist_nearest_other"] = dist_nearest_other
    df["nearest_other_gloss"] = nearest_other_gloss
    df["closer_to_other_class"] = dist_nearest_other < dist_own

    df["own_class_outlier"] = False
    for g in np.unique(glosses):
        mask = glosses == g
        if mask.sum() < MIN_CLASS_SIZE_FOR_OUTLIER_STAT:
            continue
        own_dists = dist_own[mask]
        thresh = own_dists.mean() + 3 * own_dists.std()
        df.loc[mask, "own_class_outlier"] = own_dists > thresh

    out = Path("data/silver/qa/qa_centroid_flags.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df[
        [
            "video_id",
            "gloss",
            "source",
            "dist_own",
            "dist_nearest_other",
            "nearest_other_gloss",
            "closer_to_other_class",
            "own_class_outlier",
        ]
    ].to_parquet(out, index=False)

    n_closer = df["closer_to_other_class"].sum()
    n_outlier = df["own_class_outlier"].sum()
    print(f"Total sequences: {len(df)}")
    print(f"Closer to a different class's centroid: {n_closer} ({100*n_closer/len(df):.1f}%)")
    print(f"Outlier within own class (>3 std): {n_outlier} ({100*n_outlier/len(df):.1f}%)")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/silver/keypoints/keypoints_official_split_v2norm.parquet"
    main(dataset_path)
