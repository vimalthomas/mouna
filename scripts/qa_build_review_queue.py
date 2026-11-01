#!/usr/bin/env python3
"""Build the human-review queue: full coverage of all 8,332 sequences,
grouped by gloss (word) so clips of the same claimed sign sit next to each
other -- an outlier is much easier to spot against 30 consistent neighbors
than scattered randomly through the whole set. Within each word, flagged
clips (automated signal fired) come first.

Resolves each sequence to an actual video file where one exists.
openpose_recovered has no raw video (WLASL's OpenPose-only recovery path) --
included with video_path=null, reviewable only via claimed word + flags.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd

# 95th-percentile trim-window length across our 205 WLASL100 instances that have
# a real frame_start/frame_end window (median 60, 95th pct ~123) -- an unusually
# long declared window is the proxy for "probably contains more than one sign."
LONG_WINDOW_FRAMES = 123
LONG_UNVERIFIABLE_FRAMES = 140  # openpose_recovered has no video to confirm against

DATASET = Path("data/silver/keypoints/keypoints_official_split_v2norm_aslc.parquet")
RAW_FLAGS = Path("data/silver/qa/qa_raw_flags.parquet")
CENTROID_FLAGS = Path("data/silver/qa/qa_centroid_flags.parquet")
MODEL_CONFIDENCE = Path("data/silver/qa/qa_model_confidence.parquet")
SIGNS_JSON = Path("app/backend/signs.json")
WLASL_JSON = Path("data/raw/wlasl/WLASL_v0.3.json")
OUT = Path("data/silver/qa/review_queue.json")

PROVENANCE = {
    "asl_citizen": "ASL Citizen (Microsoft Research / Boston University): signer shown a "
        "reference video of the target sign, then recorded. Project staff spot-checked "
        "samples per participant per day, not every clip.",
    "msasl_mediapipe": "MS-ASL (Microsoft): sourced from YouTube ASL content, "
        "cross-referenced against a gloss list by the dataset authors. Clip is pre-trimmed "
        "to [start_time, end_time] at download time.",
    "mediapipe_video": "WLASL benchmark: downloaded from YouTube, trimmed to WLASL's own "
        "declared frame window where one exists.",
    "openpose_recovered": "WLASL benchmark, recovered via WLASL's pre-extracted OpenPose "
        "keypoints (no raw video available to us). WLASL pre-split this before extracting "
        "keypoints, so the frame count reflects their trim, not ours -- unverifiable by eye.",
}


def load_msasl_official_paths() -> dict:
    manifest_path = Path("data/raw/msasl_official/videos/manifest.json")
    if not manifest_path.exists():
        return {}
    manifest = json.load(open(manifest_path))
    return {m["video_id"]: m["path"] for m in manifest}


def load_wlasl_trim_windows() -> dict:
    """video_id -> (frame_start, frame_end). frame_end == -1 means the WLASL
    JSON itself claims no trimming is needed (whole file is the clip)."""
    if not WLASL_JSON.exists():
        return {}
    wlasl = json.load(open(WLASL_JSON))
    windows = {}
    for entry in wlasl:
        for inst in entry.get("instances", []):
            fs = inst.get("frame_start", 1) or 1
            fe = inst.get("frame_end", -1)
            windows[str(inst["video_id"])] = (max(0, fs - 1), fe)
    return windows


_fps_cache = {}


def get_video_fps(path: str) -> float:
    if path not in _fps_cache:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            num, den = out.split("/")
            _fps_cache[path] = float(num) / float(den)
        except Exception:
            _fps_cache[path] = 29.97  # WLASL's typical YouTube-rip frame rate
    return _fps_cache[path]


def resolve_video_path(video_id: str, source: str, msasl_paths: dict) -> str | None:
    if source == "mediapipe_video":
        p = Path(f"data/raw/wlasl/videos/{int(video_id):05d}.mp4")
        return str(p) if p.exists() else None
    if source == "openpose_recovered":
        return None  # no raw video for this source
    if source == "msasl_mediapipe":
        p = msasl_paths.get(video_id)
        return p if p and Path(p).exists() else None
    if source == "asl_citizen":
        p = Path(f"data/raw/asl_citizen/videos/{video_id}")
        return str(p) if p.exists() else None
    return None


def main():
    df = pd.read_parquet(DATASET)
    raw_flags = pd.read_parquet(RAW_FLAGS)
    centroid_flags = pd.read_parquet(CENTROID_FLAGS)

    raw_flag_map = {
        row["video_id"]: json.loads(row["flags"])
        for _, row in raw_flags.iterrows()
        if row["flags"] != "[]"
    }
    quality_map = {
        row["video_id"]: {"hand_detect_rate": float(row["hand_detect_rate"]), "num_frames": int(row["num_frames"])}
        for _, row in raw_flags.iterrows()
    }
    outlier_set = set(centroid_flags[centroid_flags["own_class_outlier"]]["video_id"])

    confidence_map = {}
    if MODEL_CONFIDENCE.exists():
        conf_df = pd.read_parquet(MODEL_CONFIDENCE)
        for _, row in conf_df.iterrows():
            confidence_map[row["video_id"]] = {
                "claimed_prob": float(row["claimed_prob"]),
                "top1_gloss": row["top1_gloss"],
                "top1_prob": float(row["top1_prob"]),
                "agrees": bool(row["top1_matches_claimed"]),
            }
        print(f"Loaded model confidence for {len(confidence_map)} clips")
    else:
        print("No model confidence file yet -- field will be null (run qa_train_cv_confidence.py)")

    descriptions = {s["word"]: s["description"] for s in json.load(open(SIGNS_JSON))["signs"]}
    msasl_paths = load_msasl_official_paths()
    trim_windows = load_wlasl_trim_windows()

    long_window_ids = set()
    for vid, (fs, fe) in trim_windows.items():
        if fe != -1 and (fe - fs) > LONG_WINDOW_FRAMES:
            long_window_ids.add(vid)
    long_unverifiable_ids = {
        row["video_id"]
        for _, row in raw_flags.iterrows()
        if row["source"] == "openpose_recovered" and row["num_frames"] >= LONG_UNVERIFIABLE_FRAMES
    }

    flagged_ids = set(raw_flag_map.keys()) | outlier_set | long_window_ids | long_unverifiable_ids

    queue = []
    seen_ids = set()

    def add_item(row, reason):
        vid = str(row["video_id"])
        if vid in seen_ids:
            return
        seen_ids.add(vid)
        flags = raw_flag_map.get(vid, [])
        if vid in outlier_set:
            flags = flags + ["own_class_outlier"]
        if vid in long_window_ids:
            flags = flags + ["long_trim_window"]
        if vid in long_unverifiable_ids:
            flags = flags + ["long_unverifiable_no_video"]

        video_path = resolve_video_path(vid, row["source"], msasl_paths)
        start_time, end_time = None, None
        trim_status = "pretrimmed_at_acquisition"  # msasl_mediapipe, asl_citizen default
        if row["source"] == "mediapipe_video" and video_path:
            fs, fe = trim_windows.get(vid, (0, -1))
            if fe != -1:
                fps = get_video_fps(video_path)
                start_time, end_time = fs / fps, fe / fps
                trim_status = "real_wlasl_window"
            else:
                if fs > 0:
                    fps = get_video_fps(video_path)
                    start_time = fs / fps
                trim_status = "whole_file_declared_trimmed"
        elif row["source"] == "openpose_recovered":
            trim_status = "pretrimmed_by_wlasl_unverifiable"

        q = quality_map.get(vid, {})
        queue.append(
            {
                "video_id": vid,
                "gloss": row["gloss"],
                "description": descriptions.get(row["gloss"], ""),
                "source": row["source"],
                "provenance": PROVENANCE.get(row["source"], ""),
                "video_path": video_path,
                "start_time": start_time,
                "end_time": end_time,
                "trim_status": trim_status,
                "hand_detect_rate": q.get("hand_detect_rate"),
                "num_frames": q.get("num_frames"),
                "confidence": confidence_map.get(vid),
                "flags": flags,
                "reason": reason,
            }
        )

    df["_video_id_str"] = df["video_id"].astype(str)
    df["_is_flagged"] = df["_video_id_str"].isin(flagged_ids)
    # within each word: flagged first, then by source so same-source clips cluster too
    df = df.sort_values(by=["gloss", "_is_flagged", "source", "_video_id_str"],
                         ascending=[True, False, True, True])

    for _, row in df.iterrows():
        add_item(row, "flagged" if row["_is_flagged"] else "full_review")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(queue, open(OUT, "w"), indent=2)

    n_with_video = sum(1 for q in queue if q["video_path"])
    n_flagged = sum(1 for q in queue if q["reason"] == "flagged")
    print(f"Queue size: {len(queue)} ({n_flagged} flagged, {len(queue)-n_flagged} full_review)")
    print(f"With viewable video: {n_with_video}/{len(queue)}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
