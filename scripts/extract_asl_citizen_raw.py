#!/usr/bin/env python3
"""Extract RAW (unnormalized) keypoints from downloaded ASL Citizen clips.

Clips are already pre-trimmed to a single isolated sign (no frame-seeking
needed, unlike the WLASL YouTube clips). Saves raw flattened keypoints so
normalize_per_part can be applied uniformly at merge time, matching how the
official MS-ASL raw pool is built (see build_official_split_v2norm.py).

Parallelized across processes since MediaPipe extraction is CPU-bound and
this is ~4x our previous largest single-source extraction job.
"""

import json
import multiprocessing as mp
import pickle
from pathlib import Path

import pandas as pd

MANIFEST = Path("data/raw/asl_citizen/manifest.json")
VIDEOS_DIR = Path("data/raw/asl_citizen/videos")
OUTPUT_PARQUET = Path("data/silver/keypoints/keypoints_asl_citizen_raw.parquet")
MAX_FRAMES = 150
NUM_WORKERS = 6  # leave headroom on an 8-core machine

_extractor = None


def _init_worker():
    global _extractor
    import sys

    sys.path.insert(0, "src")
    from mouna.data.preprocessing import KeypointExtractor

    _extractor = KeypointExtractor()


def _process(entry):
    video_path = VIDEOS_DIR / entry["video_file"]
    try:
        kp = _extractor.extract_from_video(str(video_path), max_frames=MAX_FRAMES)
        flat = _extractor.flatten_keypoints(kp)
        if len(flat) == 0:
            raise ValueError("zero frames extracted")
        return {
            "video_id": entry["video_file"],
            "gloss": entry["word"],
            "keypoints_pickle": pickle.dumps(flat),
            "num_frames": len(flat),
            "success": True,
        }
    except Exception as e:
        return {
            "video_id": entry["video_file"],
            "gloss": entry["word"],
            "keypoints_pickle": None,
            "num_frames": 0,
            "success": False,
            "error": str(e),
        }


def main():
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(MANIFEST))
    print(f"Processing {len(manifest)} ASL Citizen videos with {NUM_WORKERS} workers...")

    results = []
    with mp.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(_process, manifest, chunksize=8), 1):
            results.append(res)
            if i % 200 == 0 or i == len(manifest):
                ok = sum(r["success"] for r in results)
                print(f"{i}/{len(manifest)} done ({ok} ok, {i - ok} err)")

    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    ok_df = df[df["success"]]
    print(f"\nExtracted {len(ok_df)}/{len(df)} sequences -> {OUTPUT_PARQUET}")
    print(f"Classes: {ok_df['gloss'].nunique()}")
    print(ok_df["gloss"].value_counts().to_string())

    err_df = df[~df["success"]]
    if len(err_df):
        print(f"\n{len(err_df)} errors, sample:")
        print(err_df[["video_id", "error"]].head(10).to_string())


if __name__ == "__main__":
    main()
