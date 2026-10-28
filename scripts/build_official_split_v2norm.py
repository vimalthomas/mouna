#!/usr/bin/env python3
"""Build the official-WLASL-split dataset with the new per-part normalization
(body + separate hand frames, missing-keypoint imputation) applied uniformly
across all three data sources: WLASL MediaPipe, WLASL OpenPose-recovered, and
official MS-ASL. Replaces keypoints_official_split.parquet (which still used
the old per-sequence z-score) with keypoints_official_split_v2norm.parquet.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mouna.data.openpose_convert import convert_video as op_convert_video  # noqa: E402
from mouna.data.part_normalize import normalize_per_part  # noqa: E402

ASL100_SPLIT = Path("data/raw/wlasl_extra/splits/splits/asl100.json")
RAW_MEDIAPIPE = Path("data/silver/keypoints/keypoints_raw.parquet")
RAW_MSASL = Path("data/silver/keypoints/keypoints_msasl_raw.parquet")
OUTPUT = Path("data/silver/keypoints/keypoints_official_split_v2norm.parquet")


def main():
    asl100 = json.load(open(ASL100_SPLIT))
    official_split = {}
    video_to_gloss = {}
    for g in asl100:
        for inst in g["instances"]:
            vid = str(inst["video_id"])
            official_split[vid] = inst["split"]
            video_to_gloss[vid] = g["gloss"]

    rows = []

    # 1) WLASL MediaPipe (raw, numeric ids)
    raw_mp = pd.read_parquet(RAW_MEDIAPIPE)
    raw_mp = raw_mp[raw_mp["success"]]
    raw_mp_wlasl = raw_mp[raw_mp["video_id"].astype(str).str.match(r"^\d+$")]
    have_mp_ids = set(raw_mp_wlasl["video_id"].astype(str))
    for _, row in raw_mp_wlasl.iterrows():
        vid = str(row["video_id"])
        flat = pickle.loads(row["keypoints_pickle"])
        normed = normalize_per_part(flat)
        rows.append({
            "video_id": vid, "gloss": row["gloss"],
            "keypoints_pickle": pickle.dumps(normed), "num_frames": len(normed),
            "success": True, "official_split": official_split.get(vid, "train"),
            "source": "mediapipe_video",
        })
    print(f"WLASL MediaPipe: {len(raw_mp_wlasl)} sequences")

    # 2) WLASL OpenPose-recovered (raw via convert_video, skip the old normalize_keypoints)
    pose_dir = Path("data/raw/wlasl_extra/pose_keypoints/pose_per_individual_videos")
    all_wlasl_ids = set(video_to_gloss.keys())
    available_pose_ids = set(p.name for p in pose_dir.iterdir() if p.is_dir())
    recover_ids = (all_wlasl_ids & available_pose_ids) - have_mp_ids
    ok, err = 0, 0
    for vid in sorted(recover_ids):
        try:
            flat = op_convert_video(vid)
            normed = normalize_per_part(flat)
            if len(normed) == 0:
                raise ValueError("zero frames")
            rows.append({
                "video_id": vid, "gloss": video_to_gloss[vid],
                "keypoints_pickle": pickle.dumps(normed), "num_frames": len(normed),
                "success": True, "official_split": official_split.get(vid, "train"),
                "source": "openpose_recovered",
            })
            ok += 1
        except Exception:
            err += 1
    print(f"WLASL OpenPose-recovered: {ok} ok, {err} err")

    # 3) Official MS-ASL (raw) — always train, no official WLASL split label
    raw_msasl = pd.read_parquet(RAW_MSASL)
    raw_msasl = raw_msasl[raw_msasl["success"]]
    for _, row in raw_msasl.iterrows():
        flat = pickle.loads(row["keypoints_pickle"])
        normed = normalize_per_part(flat)
        rows.append({
            "video_id": row["video_id"], "gloss": row["gloss"],
            "keypoints_pickle": pickle.dumps(normed), "num_frames": len(normed),
            "success": True, "official_split": "train",
            "source": "msasl_mediapipe",
        })
    print(f"MS-ASL: {len(raw_msasl)} sequences")

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT, index=False)
    print(f"\nTotal: {len(df)} sequences, {df['gloss'].nunique()} classes -> {OUTPUT}")
    print(df["official_split"].value_counts())


if __name__ == "__main__":
    main()
