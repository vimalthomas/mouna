#!/usr/bin/env python3
"""Build the unlabeled pretraining pool: convert ALL available WLASL OpenPose
keypoint folders (not just our 100-sign subset — pretraining needs no gloss
labels) into the same 225-dim MediaPipe-shaped, normalized format used for
fine-tuning.

Output uses the SAME schema as the labeled parquets (video_id, gloss,
keypoints_pickle, num_frames, success) with a dummy constant gloss, so
scripts/pretrain_transformer.py can reuse KeypointDataset from
scripts/train_transformer.py completely unmodified.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mouna.data.openpose_convert import convert_video, normalize_keypoints  # noqa: E402

POSE_DIR = Path(os.environ.get(
    "MOUNA_PRETRAIN_POSE_DIR",
    "data/raw/wlasl_extra/pose_keypoints/pose_per_individual_videos",
))
OUTPUT_PARQUET = Path(os.environ.get(
    "MOUNA_PRETRAIN_PARQUET", "data/silver/keypoints/pretrain_pool.parquet",
))
MAX_VIDEOS = os.environ.get("MOUNA_PRETRAIN_MAX_VIDEOS")
MAX_VIDEOS = int(MAX_VIDEOS) if MAX_VIDEOS else None
SEED = int(os.environ.get("MOUNA_PRETRAIN_SEED", 42))

DUMMY_GLOSS = "unlabeled"  # pretraining ignores this column entirely


def main():
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    video_ids = sorted(p.name for p in POSE_DIR.iterdir() if p.is_dir())
    print(f"Found {len(video_ids)} video folders under {POSE_DIR}")

    if MAX_VIDEOS and MAX_VIDEOS < len(video_ids):
        rng = np.random.RandomState(SEED)
        video_ids = list(rng.choice(video_ids, size=MAX_VIDEOS, replace=False))
        print(f"Subsampled to {len(video_ids)} (MOUNA_PRETRAIN_MAX_VIDEOS={MAX_VIDEOS}, seed={SEED})")

    # Per-sequence z-score normalization divides each feature column by its
    # own std across that sequence's frames. On short clips (few frames to
    # estimate std from), one frame that differs even slightly from the rest
    # can produce a tiny std and an enormous z-score — confirmed on this pool:
    # 33/21095 sequences (all 25-27 frames) hit |value| up to ~9155, which
    # squared in MSE corrupts an entire training batch's loss. Reject them
    # rather than silently feed them into pretraining.
    OUTLIER_MAX_ABS = 50.0

    results = []
    ok, err, outlier = 0, 0, 0
    for i, video_id in enumerate(video_ids):
        try:
            flat = convert_video(video_id, pose_dir=POSE_DIR)
            normed = normalize_keypoints(flat)
            if len(normed) == 0:
                raise ValueError("zero frames")
            max_abs = float(np.abs(normed).max())
            if max_abs > OUTLIER_MAX_ABS:
                raise ValueError(f"normalization outlier, max_abs={max_abs:.1f}")
            results.append({
                "video_id": video_id, "gloss": DUMMY_GLOSS,
                "keypoints_pickle": pickle.dumps(normed),
                "num_frames": len(normed), "success": True,
            })
            ok += 1
        except Exception as e:
            if "outlier" in str(e):
                outlier += 1
            results.append({
                "video_id": video_id, "gloss": DUMMY_GLOSS, "keypoints_pickle": None,
                "num_frames": 0, "success": False,
            })
            err += 1

        if (i + 1) % 1000 == 0:
            print(f"  progress: {i+1}/{len(video_ids)} ({ok} ok, {err} err, {outlier} outlier)")

    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nPretrain pool: {ok}/{ok+err} sequences ({outlier} rejected as normalization outliers) -> {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
