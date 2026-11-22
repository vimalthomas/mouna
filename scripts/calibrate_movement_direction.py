#!/usr/bin/env python3
"""Second attempt at a movement parameter, after calibrate_discrete_buckets.py
showed speed-tertile movement is NOT real signal (+4.7pp gap). This tries
DIRECTION/SHAPE instead of speed -- per parameters.py's docstring: "a
shape/direction-based version hasn't been built."

Same methodology as calibrate_discrete_buckets.py (bucket-agreement rate,
within-word vs across-word), reusing the same signal extraction and data
loading. Only the candidate movement classifier is new.
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

V4CANON_PATH = Path("data/silver/keypoints/keypoints_official_split_v4canon.parquet")
DECISIONS_PATH = Path("data/silver/qa/review_decisions.csv")
RAW_MEDIAPIPE = Path("data/silver/keypoints/keypoints_raw.parquet")
RAW_MSASL = Path("data/silver/keypoints/keypoints_msasl_raw.parquet")
RAW_ASLC = Path("data/silver/keypoints/keypoints_asl_citizen_raw.parquet")
MIN_CLIPS = 10
RNG_SEED = 42


def load_raw_lookup():
    lookup = {}
    for path in (RAW_MEDIAPIPE, RAW_MSASL, RAW_ASLC):
        df = pd.read_parquet(path)
        df = df[df["success"]]
        for _, row in df.iterrows():
            lookup[str(row["video_id"])] = pickle.loads(row["keypoints_pickle"])
    return lookup


def classify_movement_direction_bucket(location_traj, still_cut):
    """Net displacement (end - start) of the dominant hand in the body-
    relative x,y frame already used for location -- direction/shape, not
    speed. Discretized into compass-like buckets, same style as
    orientation's dominant-axis-with-sign. 'still' when net displacement
    is small relative to the path actually traveled (distinguishes a sign
    that moves back to near its start -- e.g. a tap or repeated motion --
    from one that's genuinely stationary; a pure start/end delta would
    conflate the two)."""
    if len(location_traj) < 2:
        return "still"
    net = location_traj[-1] - location_traj[0]
    net_dist = float(np.linalg.norm(net))
    if net_dist < still_cut:
        return "still"
    axis = int(np.argmax(np.abs(net)))
    sign = "+" if net[axis] >= 0 else "-"
    # x: + is toward the signer's left / - toward their right (post-canonicalization)
    # y: image-plane convention, + is DOWN, - is UP (matches MediaPipe's y-down)
    return f"{sign}{['x', 'y'][axis]}"


def main():
    rng = random.Random(RNG_SEED)
    decisions = pd.read_csv(DECISIONS_PATH)
    ok_ids = set(decisions[decisions["decision"] == "ok"]["video_id"].astype(str))
    v4 = pd.read_parquet(V4CANON_PATH)
    v4["_vid"] = v4["video_id"].astype(str)
    v4 = v4[v4["_vid"].isin(ok_ids)]
    raw_lookup = load_raw_lookup()

    print("Extracting signals...")
    by_word = defaultdict(list)
    for _, row in v4.iterrows():
        vid, word, source = row["_vid"], row["gloss"], row["source"]
        raw_flat = op_convert_video(vid) if source == "openpose_recovered" else raw_lookup.get(vid)
        if raw_flat is None:
            continue
        canon = canonicalize_handedness(raw_flat, target="right")
        sig = extract_dominant_hand_signal(canon)
        if sig is None:
            continue
        by_word[word].append(sig)

    words = [w for w, sigs in by_word.items() if len(sigs) >= MIN_CLIPS]
    print(f"Words with >={MIN_CLIPS} clips: {words}")

    # data-driven "still" cutoff: bottom tertile of net-displacement distances
    all_net = [
        float(np.linalg.norm(s["location_traj"][-1] - s["location_traj"][0]))
        for sigs in by_word.values() for s in sigs if len(s["location_traj"]) >= 2
    ]
    still_cut = float(np.percentile(all_net, 33))
    print(f"Net-displacement tertile cutoff (still vs. moving): {still_cut:.4f}")
    print(f"distribution: min={min(all_net):.4f}, max={max(all_net):.4f}, mean={np.mean(all_net):.4f}")

    for sigs in by_word.values():
        for s in sigs:
            s["movement_dir_bucket"] = classify_movement_direction_bucket(s["location_traj"], still_cut)

    counts = defaultdict(int)
    for sigs in by_word.values():
        for s in sigs:
            counts[s["movement_dir_bucket"]] += 1
    print(f"\nmovement_dir_bucket global distribution: {dict(counts)}")

    def pairs_agreement(key, max_pairs_per_word=40):
        within_match, within_total = 0, 0
        for w in words:
            sigs = by_word[w]
            idx_pairs = [(i, j) for i in range(len(sigs)) for j in range(i + 1, len(sigs))]
            rng.shuffle(idx_pairs)
            for i, j in idx_pairs[:max_pairs_per_word]:
                within_total += 1
                if sigs[i][key] == sigs[j][key]:
                    within_match += 1

        across_match, across_total = 0, 0
        tries = 0
        while across_total < within_total and tries < within_total * 20:
            tries += 1
            w1, w2 = rng.sample(words, 2)
            a, b = rng.choice(by_word[w1]), rng.choice(by_word[w2])
            across_total += 1
            if a[key] == b[key]:
                across_match += 1

        return within_match / within_total, across_match / across_total, within_total

    w_rate, a_rate, n = pairs_agreement("movement_dir_bucket")
    gap = w_rate - a_rate
    verdict = "REAL SIGNAL" if gap > 0.10 else "WEAK / NOT CONVINCING"
    print(f"\nmovement_dir_bucket: within-word agreement={w_rate:.3f}, across-word agreement={a_rate:.3f}, "
          f"n_pairs={n}, gap={gap:+.3f} -> {verdict}")


if __name__ == "__main__":
    main()
