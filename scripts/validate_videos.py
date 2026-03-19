#!/usr/bin/env python3
"""Validate all downloaded videos and the keypoints parquet."""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MANIFEST = Path("data/raw/wlasl/videos/manifest.json")
PARQUET = Path("data/silver/keypoints/keypoints.parquet")


def validate_videos(manifest):
    print(f"\n{'='*50}")
    print(f"Validating {len(manifest)} videos")
    print('='*50)
    ok, bad = [], []
    for entry in manifest:
        path = Path(entry["path"])
        if not path.exists():
            bad.append((entry["video_id"], "missing"))
            continue
        cap = cv2.VideoCapture(str(path))
        readable = cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if readable and frame_count > 0:
            ok.append(entry["video_id"])
        else:
            bad.append((entry["video_id"], f"unreadable (frames={frame_count})"))

    print(f"  Valid:   {len(ok)}")
    print(f"  Invalid: {len(bad)}")
    for vid, reason in bad:
        print(f"    BAD {vid}: {reason}")
    return ok, bad


def validate_parquet():
    print(f"\n{'='*50}")
    print(f"Validating keypoints parquet")
    print('='*50)
    df = pd.read_parquet(PARQUET)
    success = df[df["success"]]
    print(f"  Total rows:    {len(df)}")
    print(f"  Successful:    {len(success)}")
    print(f"  Failed:        {len(df) - len(success)}")
    print(f"\nPer-class breakdown:")
    print(success["gloss"].value_counts().to_string())

    # Check keypoint shapes
    errors = []
    for _, row in success.iterrows():
        try:
            kp = pickle.loads(row["keypoints_pickle"])
            assert kp.ndim == 2, f"wrong ndim: {kp.ndim}"
            assert kp.shape[1] == 225, f"wrong dim: {kp.shape[1]}"
            assert kp.shape[0] > 0, "zero frames"
        except Exception as e:
            errors.append((row["video_id"], str(e)))

    if errors:
        print(f"\n  Keypoint errors ({len(errors)}):")
        for vid, err in errors:
            print(f"    {vid}: {err}")
    else:
        print(f"\n  All {len(success)} keypoint arrays are valid (shape: T×225)")

    return success


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    ok_vids, bad_vids = validate_videos(manifest)
    success_df = validate_parquet()

    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"  Videos on disk (readable): {len(ok_vids)}/{ len(manifest)}")
    print(f"  Keypoints extracted:       {len(success_df)}")
    print(f"  Classes:                   {success_df['gloss'].nunique()}")
    print(f"  Ready for training:        {'YES' if len(success_df) >= 20 else 'NO — need more data'}")


if __name__ == "__main__":
    main()
