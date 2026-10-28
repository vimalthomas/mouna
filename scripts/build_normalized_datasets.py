#!/usr/bin/env python3
"""Build two normalized training sets from the same raw (pre-normalization)
keypoints, so the normalization scheme is the ONLY thing that differs between
them — isolates that one variable the same way the v1/v2 data comparison did.

  - keypoints_norm_zscore.parquet   : current scheme (per-sequence, per-feature z-score)
  - keypoints_norm_shoulder.parquet : shoulder-width body scale + per-hand wrist-relative

Both read from keypoints_raw.parquet (built by extract_keypoints_raw.py).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PARQUET = Path("data/silver/keypoints/keypoints_raw.parquet")
OUT_ZSCORE = Path("data/silver/keypoints/keypoints_norm_zscore.parquet")
OUT_SHOULDER = Path("data/silver/keypoints/keypoints_norm_shoulder.parquet")

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12  # MediaPipe pose indices


def normalize_zscore(flat: np.ndarray) -> np.ndarray:
    """Current scheme: exactly matches KeypointExtractor.normalize_keypoints."""
    mean = np.mean(flat, axis=0, keepdims=True)
    centered = flat - mean
    std = np.std(centered, axis=0, keepdims=True) + 1e-8
    return (centered / std).astype(np.float32)


def normalize_shoulder_wrist(flat: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """CenterAndScaleNormalize, ported from AI4Bharat/OpenHands (ACL 2022) —
    openhands/datasets/pose_transforms.py — rather than a hand-derived scheme.
    Clip-level (not per-frame): compute ONE center + scale for the whole clip
    from the shoulder reference points, apply it uniformly to every landmark
    (pose and both hands alike). Center = mean shoulder midpoint across all
    frames; scale = scale_factor / mean shoulder-to-shoulder distance.
    """
    T = flat.shape[0]
    arr = flat.reshape(T, 75, 3).copy()
    detected_mask = (np.abs(arr).sum(axis=2, keepdims=True) > 1e-9)  # not in the original; see note below

    left_sh = arr[:, LEFT_SHOULDER]   # (T, 3)
    right_sh = arr[:, RIGHT_SHOULDER]
    center = np.mean((left_sh + right_sh) / 2.0, axis=0)  # (3,) — clip-level
    mean_dist = np.mean(np.linalg.norm(left_sh - right_sh, axis=1))
    scale = scale_factor / mean_dist if mean_dist > 1e-9 else 1.0

    out = (arr - center) * scale
    # Not part of the original transform, but needed here: OpenHands' own data
    # pipeline drops undetected frames/joints upstream before this step, while
    # ours represents them as all-zero. Re-zero them after centering so a
    # landmark that was never detected doesn't become a fake non-zero point.
    out *= detected_mask
    return out.reshape(T, 225).astype(np.float32)


def main():
    df = pd.read_parquet(RAW_PARQUET)
    df = df[df["success"]].reset_index(drop=True)
    print(f"Raw sequences: {len(df)}")

    z_rows, s_rows = [], []
    for _, row in df.iterrows():
        flat = pickle.loads(row["keypoints_pickle"])
        z_rows.append({
            "video_id": row["video_id"], "gloss": row["gloss"],
            "keypoints_pickle": pickle.dumps(normalize_zscore(flat)),
            "num_frames": len(flat), "success": True,
        })
        s_rows.append({
            "video_id": row["video_id"], "gloss": row["gloss"],
            "keypoints_pickle": pickle.dumps(normalize_shoulder_wrist(flat)),
            "num_frames": len(flat), "success": True,
        })

    pd.DataFrame(z_rows).to_parquet(OUT_ZSCORE, index=False)
    pd.DataFrame(s_rows).to_parquet(OUT_SHOULDER, index=False)
    print(f"-> {OUT_ZSCORE} ({len(z_rows)} rows)")
    print(f"-> {OUT_SHOULDER} ({len(s_rows)} rows)")


if __name__ == "__main__":
    main()
