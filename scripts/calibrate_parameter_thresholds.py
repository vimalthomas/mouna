#!/usr/bin/env python3
"""Empirically calibrate parameters.py's GOOD/WRONG distance thresholds
against the 555 confirmed-ok clips across our 20 currently-covered words --
same methodology already used for compare.py's embedding-similarity
thresholds: compare within-word pairs (same sign, different performers) vs.
across-word pairs (different signs), and place the GOOD/WRONG boundaries in
the empirical gap between those two distributions, not guessed constants.
"""

import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from mouna.coach.parameters import extract_dominant_hand_signal  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.data.openpose_convert import convert_video as op_convert_video  # noqa: E402


def _dtw_distance(a, b):
    """Classic O(n*m) DTW, inlined here since parameters.py dropped the
    continuous-distance code this script was validating (it failed -- see
    this script's own historical output). Kept so this script still runs
    standalone as a record of that result."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        row_cost = np.linalg.norm(b - a[i - 1], axis=-1)
        for j in range(1, m + 1):
            cost[i, j] = row_cost[j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))

V4CANON_PATH = Path("data/silver/keypoints/keypoints_official_split_v4canon.parquet")
DECISIONS_PATH = Path("data/silver/qa/review_decisions.csv")
RAW_MEDIAPIPE = Path("data/silver/keypoints/keypoints_raw.parquet")
RAW_MSASL = Path("data/silver/keypoints/keypoints_msasl_raw.parquet")
RAW_ASLC = Path("data/silver/keypoints/keypoints_asl_citizen_raw.parquet")

MAX_PAIRS_PER_WORD = 40  # bounds within-word pair count; DTW is the expensive metric
RNG_SEED = 42


def load_raw_lookup():
    lookup = {}
    for path in (RAW_MEDIAPIPE, RAW_MSASL, RAW_ASLC):
        df = pd.read_parquet(path)
        df = df[df["success"]]
        for _, row in df.iterrows():
            lookup[str(row["video_id"])] = pickle.loads(row["keypoints_pickle"])
    return lookup


def main():
    rng = random.Random(RNG_SEED)

    decisions = pd.read_csv(DECISIONS_PATH)
    ok_ids = set(decisions[decisions["decision"] == "ok"]["video_id"].astype(str))

    v4 = pd.read_parquet(V4CANON_PATH)
    v4["_vid"] = v4["video_id"].astype(str)
    v4 = v4[v4["_vid"].isin(ok_ids)]

    raw_lookup = load_raw_lookup()

    print("Extracting geometric signals for all confirmed-ok clips...")
    by_word = defaultdict(list)
    skipped = 0
    for _, row in v4.iterrows():
        vid, word, source = row["_vid"], row["gloss"], row["source"]
        raw_flat = op_convert_video(vid) if source == "openpose_recovered" else raw_lookup.get(vid)
        if raw_flat is None:
            skipped += 1
            continue
        canon = canonicalize_handedness(raw_flat, target="right")
        signal = extract_dominant_hand_signal(canon)
        if signal is None:
            skipped += 1
            continue
        by_word[word].append(signal)

    print(f"Extracted {sum(len(v) for v in by_word.values())} signals, skipped {skipped}")
    for w, sigs in sorted(by_word.items()):
        print(f"  {w}: {len(sigs)}")

    def metrics(a, b):
        handshape = float(np.linalg.norm(a["handshape_curl"].mean(axis=0) - b["handshape_curl"].mean(axis=0)))
        location = float(np.linalg.norm(a["location_traj"].mean(axis=0) - b["location_traj"].mean(axis=0)))
        a_dir = a["orientation_traj"].mean(axis=0)
        b_dir = b["orientation_traj"].mean(axis=0)
        a_dir = a_dir / max(np.linalg.norm(a_dir), 1e-6)
        b_dir = b_dir / max(np.linalg.norm(b_dir), 1e-6)
        orientation = float(np.degrees(np.arccos(np.clip(np.dot(a_dir, b_dir), -1.0, 1.0))))
        movement = _dtw_distance(a["location_traj"], b["location_traj"])
        return handshape, location, orientation, movement

    within = {"handshape": [], "location": [], "orientation": [], "movement": []}
    MIN_CLIPS = 10  # exclude degenerate 2-3 clip words -- too few to say anything, just noise
    words = [w for w, sigs in by_word.items() if len(sigs) >= MIN_CLIPS]
    print(f"\nRestricting to {len(words)} words with >={MIN_CLIPS} clips: {words}")
    for w in words:
        sigs = by_word[w]
        pairs = [(i, j) for i in range(len(sigs)) for j in range(i + 1, len(sigs))]
        rng.shuffle(pairs)
        for i, j in pairs[:MAX_PAIRS_PER_WORD]:
            h, l, o, m = metrics(sigs[i], sigs[j])
            within["handshape"].append(h)
            within["location"].append(l)
            within["orientation"].append(o)
            within["movement"].append(m)

    n_within = len(within["handshape"])
    print(f"\nWithin-word pairs: {n_within}")

    across = {"handshape": [], "location": [], "orientation": [], "movement": []}
    tries = 0
    while len(across["handshape"]) < n_within and tries < n_within * 20:
        tries += 1
        w1, w2 = rng.sample(words, 2)
        a = rng.choice(by_word[w1])
        b = rng.choice(by_word[w2])
        h, l, o, m = metrics(a, b)
        across["handshape"].append(h)
        across["location"].append(l)
        across["orientation"].append(o)
        across["movement"].append(m)

    print(f"Across-word pairs: {len(across['handshape'])}")

    print(f"\n{'metric':<12} {'within mean':>12} {'within p75':>12} {'across mean':>12} {'across p25':>12}")
    suggested = {}
    for key in ("handshape", "location", "orientation", "movement"):
        w_arr, a_arr = np.array(within[key]), np.array(across[key])
        w_mean, w_p75 = w_arr.mean(), np.percentile(w_arr, 75)
        a_mean, a_p25 = a_arr.mean(), np.percentile(a_arr, 25)
        print(f"{key:<12} {w_mean:12.3f} {w_p75:12.3f} {a_mean:12.3f} {a_p25:12.3f}")
        suggested[key] = {"good_max": round(float(w_p75), 3), "wrong_min": round(float(a_p25), 3)}

    print("\nSuggested thresholds (within-word p75 / across-word p25):")
    for key, v in suggested.items():
        print(f"  {key}: good_max={v['good_max']}, wrong_min={v['wrong_min']}")
        if v["wrong_min"] <= v["good_max"]:
            print(f"    WARNING: no clean gap for {key} -- distributions overlap, treat this parameter's "
                  f"scoring as weak/exploratory, not validated.")


if __name__ == "__main__":
    main()
