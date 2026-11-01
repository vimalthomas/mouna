#!/usr/bin/env python3
"""Validate DISCRETE bucket classifiers (handshape: open/half/closed,
location: face-region/chest-region/neutral-space) as a more robust
alternative to the continuous geometric distances that just failed
calibration (calibrate_parameter_thresholds.py -- no clean within/across
gap for any of the 4 metrics).

Methodology: bucket-agreement rate, not a distance-distribution gap.
For within-word clip pairs, what fraction land in the SAME bucket? For
across-word pairs, what fraction? If within-word agreement is clearly
higher, the bucketing carries real signal; if not, this fails too and the
ensemble idea needs a different cut, not just coarser buckets.
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

NOSE = 0


def load_raw_lookup():
    lookup = {}
    for path in (RAW_MEDIAPIPE, RAW_MSASL, RAW_ASLC):
        df = pd.read_parquet(path)
        df = df[df["success"]]
        for _, row in df.iterrows():
            lookup[str(row["video_id"])] = pickle.loads(row["keypoints_pickle"])
    return lookup


def extract_with_face_anchors(canon_flat):
    """Like extract_dominant_hand_signal, but also returns nose position in
    the same body-relative frame, needed for the location-bucket anchors."""
    signal = extract_dominant_hand_signal(canon_flat)
    if signal is None:
        return None
    T = canon_flat.shape[0]
    arr = canon_flat.reshape(T, 75, 3).astype(np.float32)
    pose = arr[:, :33]
    mid_shoulder = (pose[:, 11] + pose[:, 12]) / 2.0
    shoulder_dist = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1)
    valid = shoulder_dist[shoulder_dist > 1e-6]
    scale = float(np.mean(valid)) if len(valid) else 1.0
    scale = scale if scale > 1e-6 else 1.0
    # x,y only -- see parameters.py's location_traj comment: nose z (pose model)
    # and hand z (hand model) aren't on a comparable scale.
    nose_rel = ((pose[:, NOSE] - mid_shoulder) / scale)[:, :2]  # (T, 2), body-relative, same frame as location_traj
    signal["nose_rel_mean"] = nose_rel.mean(axis=0)
    return signal


def classify_handshape_bucket(curl_mean_5dim, low_cut, high_cut):
    openness = float(np.mean(curl_mean_5dim))
    if openness < low_cut:
        return "closed"
    if openness < high_cut:
        return "half"
    return "open"


def classify_location_bucket(location_mean_3dim, nose_rel):
    """location_mean_3dim: dominant hand's mean body-relative position.
    nose_rel: nose's mean body-relative position (face anchor; chest anchor
    is the body-frame origin, mid_shoulder, by construction == (0,0,0))."""
    chest_anchor = np.zeros(2)
    d_face = float(np.linalg.norm(location_mean_3dim - nose_rel))
    d_chest = float(np.linalg.norm(location_mean_3dim - chest_anchor))
    NEUTRAL_MIN_DIST = 0.9  # beyond this from BOTH anchors -> neither face nor chest
    if d_face > NEUTRAL_MIN_DIST and d_chest > NEUTRAL_MIN_DIST:
        return "neutral_space"
    return "face_region" if d_face < d_chest else "chest_region"


def classify_movement_bucket(location_traj, low_cut, high_cut):
    """Total 2D path length per frame (average frame-to-frame displacement) --
    a coarse 'how much does the hand move' signal, not trajectory shape."""
    if len(location_traj) < 2:
        return "low"
    path_len = float(np.linalg.norm(np.diff(location_traj, axis=0), axis=-1).sum())
    speed = path_len / len(location_traj)
    if speed < low_cut:
        return "low"
    if speed < high_cut:
        return "medium"
    return "high"


def classify_orientation_bucket(orientation_mean_3dim):
    """Dominant axis (with sign) of the mean palm-normal vector -- a coarse
    'which way is the palm roughly facing' signal, not the exact angle."""
    axis = int(np.argmax(np.abs(orientation_mean_3dim)))
    sign = "+" if orientation_mean_3dim[axis] >= 0 else "-"
    return f"{sign}{['x', 'y', 'z'][axis]}"


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
        sig = extract_with_face_anchors(canon)
        if sig is None:
            continue
        by_word[word].append(sig)

    words = [w for w, sigs in by_word.items() if len(sigs) >= MIN_CLIPS]
    print(f"Words with >={MIN_CLIPS} clips: {words}")

    # --- derive global handshape openness cutoffs from the data's own distribution (tertiles) ---
    all_openness = [float(np.mean(s["handshape_curl"].mean(axis=0))) for sigs in by_word.values() for s in sigs]
    low_cut, high_cut = np.percentile(all_openness, [33, 66])
    print(f"\nHandshape openness tertile cutoffs: low={low_cut:.3f}, high={high_cut:.3f}")
    print(f"Openness distribution: min={min(all_openness):.3f}, max={max(all_openness):.3f}, "
          f"mean={np.mean(all_openness):.3f}")

    # --- movement speed tertile cutoffs, same data-driven approach ---
    all_speed = [
        float(np.linalg.norm(np.diff(s["location_traj"], axis=0), axis=-1).sum()) / max(len(s["location_traj"]), 1)
        for sigs in by_word.values() for s in sigs
    ]
    speed_low, speed_high = np.percentile(all_speed, [33, 66])
    print(f"Movement speed tertile cutoffs: low={speed_low:.4f}, high={speed_high:.4f}")

    # --- bucket every clip ---
    for sigs in by_word.values():
        for s in sigs:
            s["handshape_bucket"] = classify_handshape_bucket(s["handshape_curl"].mean(axis=0), low_cut, high_cut)
            s["location_bucket"] = classify_location_bucket(s["location_traj"].mean(axis=0), s["nose_rel_mean"])
            s["movement_bucket"] = classify_movement_bucket(s["location_traj"], speed_low, speed_high)
            s["orientation_bucket"] = classify_orientation_bucket(s["orientation_traj"].mean(axis=0))

    for key in ("handshape_bucket", "location_bucket", "movement_bucket", "orientation_bucket"):
        counts = defaultdict(int)
        for sigs in by_word.values():
            for s in sigs:
                counts[s[key]] += 1
        print(f"\n{key} global distribution: {dict(counts)}")

    # --- within vs across agreement rate ---
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

    for key in ("handshape_bucket", "location_bucket", "movement_bucket", "orientation_bucket"):
        w_rate, a_rate, n = pairs_agreement(key)
        gap = w_rate - a_rate
        verdict = "REAL SIGNAL" if gap > 0.10 else "WEAK / NOT CONVINCING"
        print(f"\n{key}: within-word agreement={w_rate:.3f}, across-word agreement={a_rate:.3f}, "
              f"n_pairs={n}, gap={gap:+.3f} -> {verdict}")


if __name__ == "__main__":
    main()
