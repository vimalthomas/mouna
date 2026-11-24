#!/usr/bin/env python3
"""Pre-render the two fluent-signer rows of the LLM hand-count check's image
for every sign with an example video and a second approved clip, using the
benchmark's own frame picking (llm_vision_judge.key_frames) and the same two
references it used (the example video, then the first other approved clip).

Writes data/references/llm_refs/<sign>/row1.jpg, row2.jpg
Run from repo root: ./venv/bin/python scripts/build_llm_reference_rows.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
_argv, sys.argv = sys.argv, sys.argv[:1]
import llm_vision_judge as V  # noqa: E402
sys.argv = _argv
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.llm_judge import REF_DIR  # noqa: E402


def main() -> None:
    raw = load_raw_lookup()
    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    approved = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss"])
    v4["vid"] = v4.video_id.astype(str)
    made, skipped = 0, []
    for g, sub in v4[v4.vid.isin(approved)].groupby("gloss"):
        if g not in V.EXAMPLES:
            continue
        tvid = V.EXAMPLES[g]["video_id"]
        tpath = ROOT / f"app/wlasl/videos/{tvid}.mp4"
        second = next((v for v in sub.vid if v != tvid and V.video_path(v) and v in raw), None)
        if not tpath.exists() or second is None:
            skipped.append(g)
            continue
        rows = []
        for p, vid in ((tpath, tvid), (V.video_path(second), second)):
            fs = V.key_frames(p, V.track(vid, raw, None))
            rows.append(np.hstack(fs) if len(fs) >= 2 else None)
        if any(r is None for r in rows):
            skipped.append(g)
            continue
        d = REF_DIR / g
        d.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows, 1):
            cv2.imwrite(str(d / f"row{i}.jpg"), r, [cv2.IMWRITE_JPEG_QUALITY, 92])
        made += 1
    print(f"{made} signs -> {REF_DIR.relative_to(ROOT)}; skipped: {skipped}")


if __name__ == "__main__":
    main()
