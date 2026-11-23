#!/usr/bin/env python3
"""Recalibrate the LOCATION bucket against an independent, expert-coded
answer key: ASL-LEX 2.0 `MajorLocation.2.0` per sign.

Why: the current bucket (mean wrist position vs a "chest" anchor at the
shoulder line) put 0 of 9 face/head signs in face_region, so its region
names are hidden from learners. Here several principled variants are scored
on every human-approved clip of every sign ASL-LEX codes, with a held-out
check: variants are ranked on half the signs (tune) and the winner is
reported on the other half (test), so the choice isn't fit to one list.

ASL-LEX MajorLocation -> Mouna bucket:
  Head -> face_region; Body -> chest_region; Neutral -> neutral_space.
  Hand/Arm/Other are skipped (location is relative to the other hand/arm).

Run from repo root:  ./venv/bin/python scripts/calibrate_location_asllex.py
"""
import csv
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = ["src", "scripts"]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.data.part_normalize import _interpolate_missing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"Head": "face_region", "Body": "chest_region", "Neutral": "neutral_space"}


def asl_lex_locations() -> dict[str, str]:
    out = {}
    for row in csv.DictReader(open(ROOT / "data/external/asl_lex/signdata.csv", encoding="utf-8", errors="replace")):
        w = re.sub(r"_\d+$", "", row["EntryID"].lower())
        loc = LABEL.get(row.get("MajorLocation.2.0", ""))
        if loc and w not in out:
            out[w] = loc
    return out


def features(canon: np.ndarray):
    """Per-frame dominant-hand points + face/body anchors, body-relative."""
    a = canon.reshape(len(canon), 75, 3)
    hand = a[:, 54:75]
    present = np.any(hand != 0, axis=(1, 2))
    if present.sum() < 3:
        return None
    pose = _interpolate_missing(a[:, :33])
    mid = (pose[:, 11] + pose[:, 12]) / 2
    sc = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1)
    sc = float(sc[sc > 1e-6].mean()) if (sc > 1e-6).any() else 1.0
    h = _interpolate_missing(hand)
    rel = lambda p: ((p - mid) / sc)[:, :2]  # noqa: E731
    return {
        "wrist": rel(h[:, 0])[present],
        "center": rel(h[:, [0, 5, 9, 13, 17]].mean(1))[present],
        "nose": rel(pose[:, 0])[present].mean(0),
    }


def classify(f, point, agg, chest_y, neutral):
    tr = f[point]
    if agg == "mean":
        loc = tr.mean(0)
    else:  # mean of the frames where the hand is highest (sign's core)
        k = max(1, int(len(tr) * agg))
        loc = tr[np.argsort(tr[:, 1])[:k]].mean(0)
    d_face = np.linalg.norm(loc - f["nose"])
    d_chest = np.linalg.norm(loc - np.array([0.0, chest_y]))
    if d_face > neutral and d_chest > neutral:
        return "neutral_space"
    return "face_region" if d_face < d_chest else "chest_region"


VARIANTS = list(itertools.product(["wrist", "center"], ["mean", 0.5, 0.3], [0.0, 0.4, 0.6], [0.9, 1.2]))


def main() -> None:
    truth = asl_lex_locations()
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss"])
    v4["vid"] = v4.video_id.astype(str)
    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    ok = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = v4[v4.vid.isin(ok) & v4.gloss.isin(truth)]
    raw = load_raw_lookup()

    clips = defaultdict(list)
    for _, r in v4.iterrows():
        flat = raw.get(r.vid)
        if flat is None:
            continue
        f = features(canonicalize_handedness(np.asarray(flat, dtype=np.float32), target="right"))
        if f is not None:
            clips[r.gloss].append(f)
    signs = sorted(clips)
    print(f"{sum(len(v) for v in clips.values())} approved clips, {len(signs)} signs with an ASL-LEX location")
    print("label counts:", dict(pd.Series([truth[s] for s in signs]).value_counts()))

    def sign_accuracy(variant, subset):
        # per-sign majority vote of its clips, as the app's targets work now
        hits = 0
        for s in subset:
            votes = pd.Series([classify(f, *variant) for f in clips[s]]).value_counts()
            hits += votes.index[0] == truth[s]
        return hits / len(subset)

    random.seed(7)
    order = signs[:]
    random.shuffle(order)
    tune, test = order[: len(order) // 2], order[len(order) // 2:]
    current = ("wrist", "mean", 0.0, 0.9)
    ranked = sorted(VARIANTS, key=lambda v: sign_accuracy(v, tune), reverse=True)
    best = ranked[0]
    result = {
        "current": {"variant": current, "tune": sign_accuracy(current, tune), "test": sign_accuracy(current, test), "all": sign_accuracy(current, signs)},
        "best": {"variant": best, "tune": sign_accuracy(best, tune), "test": sign_accuracy(best, test), "all": sign_accuracy(best, signs)},
        "n_signs": {"tune": len(tune), "test": len(test)},
    }
    print(json.dumps(result, indent=2, default=str))
    print("\nper-sign with best variant (all signs):")
    for s in signs:
        votes = pd.Series([classify(f, *best) for f in clips[s]]).value_counts()
        mark = "ok " if votes.index[0] == truth[s] else "MISS"
        print(f"  {mark} {s:11s} truth={truth[s]:13s} got={votes.index[0]:13s} ({votes.iloc[0]}/{len(clips[s])})")
    (ROOT / "data/references/location_calibration.json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
