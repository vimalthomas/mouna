#!/usr/bin/env python3
"""Extract keypoints from downloaded videos (WLASL + MS-ASL) and save as parquet.

For WLASL videos that are long YouTube clips, the WLASL JSON provides frame_start/
frame_end timestamps pointing to the actual sign segment. We seek directly to those
frames so the model trains on the correct sign, not surrounding content.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from mouna.data.preprocessing import KeypointExtractor

# All manifest files to combine. Overridable via env (comma-separated) so this
# script can also run against a single new manifest (e.g. the official MS-ASL
# re-download) without touching the baseline keypoints.parquet.
_manifests_env = os.environ.get("MOUNA_EXTRACT_MANIFESTS")
MANIFESTS = (
    [Path(p) for p in _manifests_env.split(",")] if _manifests_env else
    [
        Path("data/raw/wlasl/videos/manifest.json"),
        Path("data/raw/msasl/videos/manifest.json"),  # optional — skipped if absent
    ]
)
WLASL_JSON     = Path("data/raw/wlasl/WLASL_v0.3.json")
OUTPUT_PARQUET = Path(os.environ.get("MOUNA_EXTRACT_OUTPUT", "data/silver/keypoints/keypoints.parquet"))
MAX_FRAMES     = 150  # cap on frames extracted per clip


def _build_timestamp_index(wlasl_json: Path) -> dict:
    """Return {video_id: (start_frame, end_frame)} from WLASL_v0.3.json.

    frame_end == -1 means the file is already a trimmed clip (start=0, end=-1).
    frame_start is 1-based in the WLASL JSON; we convert to 0-based here.
    """
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
            index[vid_id] = (max(0, fs - 1), fe)  # 0-based start
    return index


def main():
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    # Build timestamp lookup from WLASL JSON
    ts_index = _build_timestamp_index(WLASL_JSON)
    trimmed_count = sum(1 for (s, e) in ts_index.values() if s > 0)
    print(f"Timestamp index: {len(ts_index)} entries, {trimmed_count} need frame-seeking")

    manifest = []
    for m in MANIFESTS:
        if not m.exists():
            print(f"Skipping missing manifest: {m}")
            continue
        with open(m) as f:
            entries = json.load(f)
        print(f"Loaded {len(entries)} entries from {m}")
        manifest.extend(entries)

    # Deduplicate by video_id
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

    for entry in manifest:
        video_id    = str(entry["video_id"])
        gloss       = entry["gloss"]
        path        = entry["path"]
        start_frame, end_frame = ts_index.get(video_id, (0, -1))

        try:
            kp = extractor.extract_from_video(
                path,
                max_frames=MAX_FRAMES,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            flat   = extractor.flatten_keypoints(kp)
            normed = extractor.normalize_keypoints(flat)

            if len(normed) == 0:
                raise ValueError("zero frames extracted")

            results.append({
                "video_id":         video_id,
                "gloss":            gloss,
                "keypoints_pickle": pickle.dumps(normed),
                "num_frames":       len(normed),
                "success":          True,
            })
            seek_info = f" [frames {start_frame}→{end_frame}]" if start_frame > 0 else ""
            print(f"  OK  {video_id} ({gloss}){seek_info} — {len(normed)} frames")
            ok += 1
        except Exception as e:
            print(f"  ERR {video_id} ({gloss}): {e}")
            results.append({
                "video_id":         video_id,
                "gloss":            gloss,
                "keypoints_pickle": None,
                "num_frames":       0,
                "success":          False,
            })
            err += 1

    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    success_df = df[df["success"]]
    print(f"\nExtracted {ok}/{ok+err} sequences")
    print(f"Saved → {OUTPUT_PARQUET}")
    print(f"Classes: {sorted(success_df['gloss'].unique())}")
    print(f"Per-class counts:\n{success_df['gloss'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
