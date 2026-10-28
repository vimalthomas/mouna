#!/usr/bin/env python3
"""Convert WLASL's released OpenPose BODY_25 keypoints to our MediaPipe-shaped
feature format, for WLASL-100 instances whose raw video we could never download
(YouTube links now dead) but whose keypoints were extracted by the WLASL authors
back in 2020, before those links died.

Output is kept as a SEPARATE parquet from the existing MediaPipe-extracted
keypoints.parquet, and a merged v2 is written alongside it — so v1 (current,
camera-extracted only) and v2 (v1 + OpenPose-recovered) can be trained and
compared independently rather than silently blending two different source
distributions into one file.

Mapping notes (see WHY in each block):
  - Pose: OpenPose BODY_25 has 25 joints, MediaPipe Pose has 33. Only the
    anatomically-shared joints (nose/eyes/ears/shoulders/elbows/wrists/hips/
    knees/ankles/heels/toes) are mapped; MediaPipe-only points with no BODY_25
    equivalent (eye corners, mouth corners, pose-level finger landmarks) are
    left zero.
  - Hands: OpenPose's 21-point hand topology (wrist + 4 joints x 4 fingers) is
    the same convention MediaPipe Hands uses, so hand landmarks map 1:1 with
    no reordering.
  - Depth: OpenPose is 2D-only (x, y, confidence). z is set to 0 for all
    converted points — a real information loss vs. true MediaPipe z, which is
    exactly why this stays a separate, labeled dataset rather than being
    blended invisibly into the original.
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from mouna.data.openpose_convert import convert_video, normalize_keypoints  # noqa: E402

SPLITS_JSON = Path("data/raw/wlasl_extra/splits/splits/asl100.json")
POSE_DIR = Path("data/raw/wlasl_extra/pose_keypoints/pose_per_individual_videos")
EXISTING_PARQUET = Path("data/silver/keypoints/keypoints.parquet")
RECOVERED_OUT = Path("data/silver/keypoints/keypoints_v2b_openpose_recovered_targeted.parquet")
MERGED_OUT = Path("data/silver/keypoints/keypoints_v2b_merged_targeted.parquet")

# Only backfill classes that are genuinely short on data. The recovered data is
# real but noisier (hand-keypoint coverage ~48% vs ~12% missing in camera-
# extracted data) — adding it to classes that already have enough samples just
# dilutes them with no corresponding benefit. Classes at/above this floor are
# left untouched.
TARGETED_CLASS_FLOOR = 20


def main():
    RECOVERED_OUT.parent.mkdir(parents=True, exist_ok=True)

    wlasl100 = json.load(open(SPLITS_JSON))
    video_to_gloss = {}
    for entry in wlasl100:
        for inst in entry["instances"]:
            video_to_gloss[str(inst["video_id"])] = entry["gloss"]

    existing_df = pd.read_parquet(EXISTING_PARQUET)
    existing_ids = set(existing_df["video_id"].astype(str))
    print(f"Existing (v1) parquet: {len(existing_df)} sequences")

    available = set(p.name for p in POSE_DIR.iterdir() if p.is_dir())
    target_ids = (set(video_to_gloss) & available) - existing_ids
    print(f"Recoverable via OpenPose (not already in v1): {len(target_ids)}")

    v1_counts = existing_df[existing_df["success"]]["gloss"].value_counts()
    thin_glosses = set(v1_counts[v1_counts < TARGETED_CLASS_FLOOR].index) | (
        set(video_to_gloss.values()) - set(v1_counts.index)  # classes with 0 v1 samples
    )
    before = len(target_ids)
    target_ids = {vid for vid in target_ids if video_to_gloss[vid] in thin_glosses}
    print(f"Targeted mode: {len(thin_glosses)} classes below floor ({TARGETED_CLASS_FLOOR}); "
          f"{len(target_ids)}/{before} recoverable instances kept (rest skipped — class already adequate)")

    results = []
    ok, err = 0, 0
    for video_id in sorted(target_ids):
        gloss = video_to_gloss[video_id]
        try:
            flat = convert_video(video_id)
            normed = normalize_keypoints(flat)
            if len(normed) == 0:
                raise ValueError("zero frames")
            results.append({
                "video_id": video_id,
                "gloss": gloss,
                "keypoints_pickle": pickle.dumps(normed),
                "num_frames": len(normed),
                "success": True,
                "source": "openpose_recovered",
            })
            ok += 1
        except Exception as e:
            results.append({
                "video_id": video_id, "gloss": gloss, "keypoints_pickle": None,
                "num_frames": 0, "success": False, "source": "openpose_recovered",
            })
            print(f"  ERR {video_id} ({gloss}): {e}")
            err += 1

    recovered_df = pd.DataFrame(results)
    recovered_df.to_parquet(RECOVERED_OUT, index=False)
    print(f"\nRecovered {ok}/{ok+err} sequences -> {RECOVERED_OUT}")

    existing_df["source"] = existing_df.get("source", "mediapipe_video")
    merged = pd.concat([existing_df, recovered_df[recovered_df["success"]]], ignore_index=True)
    merged.to_parquet(MERGED_OUT, index=False)
    print(f"v2 merged (v1 + recovered): {len(merged)} sequences -> {MERGED_OUT}")

    print("\nPer-class counts, v1 vs v2:")
    v1_counts = existing_df[existing_df["success"]]["gloss"].value_counts()
    v2_counts = merged[merged["success"]]["gloss"].value_counts()
    all_glosses = sorted(set(video_to_gloss.values()))
    comparison = pd.DataFrame({
        "v1": v1_counts.reindex(all_glosses, fill_value=0),
        "v2": v2_counts.reindex(all_glosses, fill_value=0),
    })
    comparison["gained"] = comparison["v2"] - comparison["v1"]
    comparison = comparison.sort_values("v2")
    print(comparison.to_string())
    still_thin = comparison[comparison["v2"] < 10]
    print(f"\nStill <10 samples after merge: {len(still_thin)} classes")
    print(still_thin.to_string())


if __name__ == "__main__":
    main()
