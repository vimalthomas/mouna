#!/usr/bin/env python3
"""Width x height of every dataset clip we have a video for, so tracked
points (stored as fractions of width / height) can be turned back into true
proportions before measuring angles. Writes data/references/video_sizes.json.
"""
import json
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = [ROOT / "data/raw/asl_citizen/videos", ROOT / "app/wlasl/videos", ROOT / "data/raw/msasl/videos"]


def path_for(vid: str) -> Path | None:
    for d in FOLDERS:
        for name in (vid, f"{vid}.mp4"):
            if (d / name).exists():
                return d / name
    return None


def main() -> None:
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id"])
    out = {}
    for vid in v4.video_id.astype(str).unique():
        p = path_for(vid)
        if p is None:
            continue
        cap = cv2.VideoCapture(str(p))
        w, h = int(cap.get(3)), int(cap.get(4))
        cap.release()
        if w and h:
            out[vid] = [w, h]
    (ROOT / "data/references/video_sizes.json").write_text(json.dumps(out))
    print(f"{len(out)} of {v4.video_id.nunique()} clips have a video file")


if __name__ == "__main__":
    main()
