#!/usr/bin/env python3
"""Extract RAW (pre-normalization) keypoints from WLASL+MS-ASL videos.

Mirrors extract_keypoints.py exactly, except it stores the flattened,
un-normalized 225-dim features instead of the z-scored ones. This is the
reusable source-of-truth for testing any normalization scheme (the current
per-sequence z-score, or the shoulder-width/wrist-relative alternative)
without re-running MediaPipe each time.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from mouna.data.preprocessing import KeypointExtractor

MANIFESTS = [
    Path("data/raw/wlasl/videos/manifest.json"),
    Path("data/raw/msasl/videos/manifest.json"),
]
WLASL_JSON = Path("data/raw/wlasl/WLASL_v0.3.json")
OUTPUT_PARQUET = Path("data/silver/keypoints/keypoints_raw.parquet")
MAX_FRAMES = 150


def _build_timestamp_index(wlasl_json: Path) -> dict:
    if not wlasl_json.exists():
        return {}
    with open(wlasl_json) as f:
        wlasl = json.load(f)
    index = {}
    for entry in wlasl:
        for inst in entry.get("instances", []):
            vid_id = str(inst["video_id"])
            fs = inst.get("frame_start", 1) or 1
            fe = inst.get("frame_end", -1)
            index[vid_id] = (max(0, fs - 1), fe)
    return index


def main():
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    ts_index = _build_timestamp_index(WLASL_JSON)
    print(f"Timestamp index: {len(ts_index)} entries")

    manifest = []
    for m in MANIFESTS:
        if not m.exists():
            print(f"Skipping missing manifest: {m}")
            continue
        with open(m) as f:
            entries = json.load(f)
        print(f"Loaded {len(entries)} entries from {m}")
        manifest.extend(entries)

    seen = set()
    deduped = []
    for entry in manifest:
        if entry["video_id"] not in seen:
            seen.add(entry["video_id"])
            deduped.append(entry)
    manifest = deduped

    print(f"Processing {len(manifest)} videos...")
    extractor = KeypointExtractor()

    results = []
    ok, err = 0, 0
    for i, entry in enumerate(manifest):
        video_id = str(entry["video_id"])
        gloss = entry["gloss"]
        path = entry["path"]
        start_frame, end_frame = ts_index.get(video_id, (0, -1))

        try:
            kp = extractor.extract_from_video(
                path, max_frames=MAX_FRAMES, start_frame=start_frame, end_frame=end_frame,
            )
            flat = extractor.flatten_keypoints(kp)  # RAW, not normalized
            if len(flat) == 0:
                raise ValueError("zero frames extracted")

            results.append({
                "video_id": video_id, "gloss": gloss,
                "keypoints_pickle": pickle.dumps(flat.astype(np.float32)),
                "num_frames": len(flat), "success": True,
            })
            ok += 1
        except Exception as e:
            results.append({
                "video_id": video_id, "gloss": gloss, "keypoints_pickle": None,
                "num_frames": 0, "success": False,
            })
            print(f"  ERR {video_id} ({gloss}): {e}")
            err += 1

        if (i + 1) % 100 == 0:
            print(f"  progress: {i+1}/{len(manifest)} ({ok} ok, {err} err)")

    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nExtracted {ok}/{ok+err} sequences (RAW, un-normalized) -> {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
