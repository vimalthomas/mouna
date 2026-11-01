#!/usr/bin/env python3
"""Automated QA layer 1: video-quality flags computed from RAW (pre-normalization,
pre-interpolation) keypoints, per source.

These catch corruption/extraction problems, not sign-correctness -- a hand
can be perfectly detected while the wrong sign is performed. See
qa_flag_centroid.py for the sign-plausibility layer.

Flags:
  short_clip        -- fewer than MIN_FRAMES frames (can't be a real sign)
  low_hand_detect    -- hands visible in under MIN_HAND_DETECT_RATE of frames
  duplicate          -- exact byte-identical raw keypoints as another video

Run separately per source (each has its own raw parquet / recovery path) and
concatenate; see qa_build_review_queue.py for the merge step.
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

MIN_FRAMES = 8
MIN_HAND_DETECT_RATE = 0.10

# pose=33*3=99, left_hand=21*3=63, right_hand=21*3=63 -> flat 225
POSE_DIMS = 99
HAND_DIMS = 63


def hand_detect_rate(flat: np.ndarray) -> float:
    left = flat[:, POSE_DIMS : POSE_DIMS + HAND_DIMS]
    right = flat[:, POSE_DIMS + HAND_DIMS : POSE_DIMS + 2 * HAND_DIMS]
    any_hand = np.any(left != 0, axis=1) | np.any(right != 0, axis=1)
    return float(np.mean(any_hand))


def flag_source(video_id: str, gloss: str, flat: np.ndarray, source: str) -> dict:
    n = len(flat)
    rate = hand_detect_rate(flat) if n > 0 else 0.0
    dup_hash = hashlib.md5(flat.tobytes()).hexdigest()
    flags = []
    if n < MIN_FRAMES:
        flags.append("short_clip")
    if rate < MIN_HAND_DETECT_RATE:
        flags.append("low_hand_detect")
    return {
        "video_id": video_id,
        "gloss": gloss,
        "source": source,
        "num_frames": n,
        "hand_detect_rate": rate,
        "dup_hash": dup_hash,
        "flags": flags,
    }


def from_raw_parquet(path: Path, source: str) -> list[dict]:
    df = pd.read_parquet(path)
    df = df[df["success"]]
    rows = []
    for _, row in df.iterrows():
        flat = pickle.loads(row["keypoints_pickle"])
        rows.append(flag_source(str(row["video_id"]), row["gloss"], flat, source))
    return rows


def from_openpose_recovered() -> list[dict]:
    from mouna.data.openpose_convert import convert_video as op_convert_video

    asl100 = json.load(open("data/raw/wlasl_extra/splits/splits/asl100.json"))
    official_split, video_to_gloss = {}, {}
    for g in asl100:
        for inst in g["instances"]:
            vid = str(inst["video_id"])
            official_split[vid] = inst["split"]
            video_to_gloss[vid] = g["gloss"]

    raw_mp = pd.read_parquet("data/silver/keypoints/keypoints_raw.parquet")
    raw_mp_wlasl = raw_mp[raw_mp["video_id"].astype(str).str.match(r"^\d+$")]
    have_mp_ids = set(raw_mp_wlasl["video_id"].astype(str))

    pose_dir = Path("data/raw/wlasl_extra/pose_keypoints/pose_per_individual_videos")
    available_pose_ids = set(p.name for p in pose_dir.iterdir() if p.is_dir())
    recover_ids = (set(video_to_gloss.keys()) & available_pose_ids) - have_mp_ids

    rows = []
    for vid in sorted(recover_ids):
        try:
            flat = op_convert_video(vid)
            if len(flat) == 0:
                continue
            rows.append(flag_source(vid, video_to_gloss[vid], flat, "openpose_recovered"))
        except Exception:
            continue
    return rows


def mark_duplicates(rows: list[dict]) -> None:
    by_hash = {}
    for r in rows:
        by_hash.setdefault(r["dup_hash"], []).append(r)
    for group in by_hash.values():
        if len(group) > 1:
            for r in group:
                r["flags"] = r["flags"] + ["duplicate"]
                r["duplicate_of"] = [g["video_id"] for g in group if g["video_id"] != r["video_id"]]


def main():
    all_rows = []
    all_rows += from_raw_parquet(Path("data/silver/keypoints/keypoints_raw.parquet"), "mediapipe_video")
    all_rows += from_openpose_recovered()
    all_rows += from_raw_parquet(Path("data/silver/keypoints/keypoints_msasl_raw.parquet"), "msasl_mediapipe")

    aslc_raw = Path("data/silver/keypoints/keypoints_asl_citizen_raw.parquet")
    if aslc_raw.exists():
        all_rows += from_raw_parquet(aslc_raw, "asl_citizen")
    else:
        print("ASL Citizen raw parquet not found yet -- skipping (run again once extraction completes)")

    mark_duplicates(all_rows)

    df = pd.DataFrame(all_rows)
    df["flags"] = df["flags"].apply(json.dumps)
    out = Path("data/silver/qa/qa_raw_flags.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    flagged = df[df["flags"] != "[]"]
    print(f"\nTotal sequences checked: {len(df)}")
    print(f"Flagged: {len(flagged)} ({100*len(flagged)/len(df):.1f}%)")
    print(flagged["source"].value_counts())
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
