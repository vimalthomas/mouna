#!/usr/bin/env python3
"""Quality check for newly downloaded ASL Citizen clips, before anything is
built from them. Per clip:
  - exact duplicate files (md5)
  - same signer recorded more than once for the same sign
  - duration (one isolated sign should be roughly 0.8-6 s)
  - hand tracking: share of sampled frames (10 fps, the app's rate) where
    MediaPipe finds at least one hand -- low means occlusion, framing or
    lighting problems, which also make the clip useless for references.
A clip is "keep" unless flagged. Writes data/silver/qa/new_aslc_clips_qa.csv
and prints a per-sign summary.

Run from repo root:
    ./venv/bin/python scripts/qa_new_aslc_clips.py sit,ask,dry,small,not,why,where,when
"""
import base64
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app" / "backend"), str(ROOT / "src")]
import os  # noqa: E402

os.chdir(ROOT / "app" / "backend")
from services.keypoint_service import keypoint_service  # noqa: E402

ASLC = ROOT / "data" / "raw" / "asl_citizen"
OUT = ROOT / "data" / "silver" / "qa" / "new_aslc_clips_qa.csv"
MIN_S, MAX_S = 0.8, 6.0
# Calibrated on known-good clips: ASL Citizen clips of 4 trained signs
# (who, drink, apple, mother; n=48) have a median hand share of 0.42 and a
# 10th percentile of 0.27 -- clips start and end with hands resting out of
# view. 0.5 flagged 81% of those good clips, so the bar is set just below
# their 10th percentile: only clips worse than nearly all good ones flag.
MIN_HAND_SHARE = 0.25


def check(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / 10))
    sampled = hands = i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            jpg = cv2.imencode(".jpg", cv2.resize(img, (640, 480)))[1]
            kp = np.asarray(keypoint_service.extract_from_base64(base64.b64encode(jpg).decode()))
            sampled += 1
            hands += bool(np.any(kp[99:225] != 0))
        i += 1
    return {"duration_s": round(n / fps, 2) if fps else 0, "width": w, "height": h,
            "hand_share": round(hands / sampled, 2) if sampled else 0.0}


def main() -> None:
    words = [w.strip().lower() for w in sys.argv[1].split(",")]
    keypoint_service.initialize()
    manifest = [r for r in json.load(open(ASLC / "manifest.json")) if r["word"] in words]
    md5s: dict[str, str] = {}
    per_signer = Counter((r["word"], r["participant_id"]) for r in manifest)
    rows = []
    for r in manifest:
        p = ASLC / "videos" / r["video_file"]
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        info = check(p)
        flags = []
        if digest in md5s:
            flags.append(f"duplicate_of:{md5s[digest]}")
        md5s.setdefault(digest, r["video_file"])
        if per_signer[(r["word"], r["participant_id"])] > 1:
            flags.append("signer_repeat")
        if not (MIN_S <= info["duration_s"] <= MAX_S):
            flags.append("duration")
        if info["hand_share"] < MIN_HAND_SHARE:
            flags.append("low_hand_tracking")
        # a repeat by the same signer is fine to keep (a second take), but it
        # doesn't add a NEW signer -- note it, don't drop it
        hard = [f for f in flags if f != "signer_repeat"]
        rows.append({**r, **info, "md5": digest, "flags": ";".join(flags), "keep": not hard})
        print(f"{r['word']:6s} {r['video_file'][:34]:34s} {info['duration_s']:5.2f}s hands={info['hand_share']:.2f} {';'.join(flags)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\nsign    clips keep signers(kept) flagged")
    by = defaultdict(list)
    for r in rows:
        by[r["word"]].append(r)
    for w in words:
        rs = by[w]
        kept = [r for r in rs if r["keep"]]
        bad = Counter(f.split(":")[0] for r in rs for f in r["flags"].split(";") if f and f != "signer_repeat")
        print(f"{w:7s} {len(rs):5d} {len(kept):4d} {len({r['participant_id'] for r in kept}):8d}       {dict(bad)}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
