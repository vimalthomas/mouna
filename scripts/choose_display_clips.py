#!/usr/bin/env python3
"""Pick, per sign, the approved clip whose HANDS ARE TRACKED BEST, for the
3D hand view -- separate from the clip used for grading.

Why: the reference's representative clip is the one closest to the sign's
average embedding (right for grading). With only ~10 approved clips (the
newly reviewed signs), the most "average" clip is often a jittery or partly
occluded one, and the 3D hands built from it looked worse than the first
signs (which had 43-113 approved clips to choose from). Display needs the
cleanest tracking instead.

Quality score per approved clip (higher is better):
  1. source: ASL Citizen (close, HD) > WLASL video > MS-ASL (often low-res);
     OpenPose recoveries have no depth and are never chosen
  2. finger steadiness: mean frame-to-frame jump of the dominant hand's
     hand-local 3D points while visible (jitter), lower = better
Signs in KEEP_AS_IS keep their current 3D (the first signs, which the user
likes as they are).

Writes `display_sequence_raw_canon` (+ `display_source`) into
data/references/sign_references.pkl; model_service.get_reference_skeleton
prefers it when present.

Run from repo root:  ./venv/bin/python scripts/choose_display_clips.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from build_sign_references_v2 import load_raw_lookup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "data/references/sign_references.pkl"
V4 = ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet"
DECISIONS = ROOT / "data/silver/qa/review_decisions.csv"
KEEP_AS_IS = {"accident", "africa", "all", "apple", "basketball", "bed", "before", "bird", "can",
              "color", "decide", "go", "how", "hot", "need", "no", "now", "orange", "paper",
              "short", "want", "what", "who"}
# Sharper sources first; within a source, the steadiest fingers. (Ranking by
# "share of frames with a hand" favored MS-ASL, because ASL Citizen clips
# start and end with hands at rest -- not a quality signal.)
SOURCE_RANK = {"asl_citizen": 0, "mediapipe_video": 1, "msasl_mediapipe": 2}
SOURCE_BONUS = SOURCE_RANK


def quality(canon: np.ndarray) -> tuple[float, float, float] | None:
    """(jitter, visible share) for the dominant ('right' after canonicalization) hand."""
    arr = canon.reshape(len(canon), 75, 3)
    hand = arr[:, 54:75]
    present = np.any(hand != 0, axis=(1, 2))
    if present.sum() < 5:
        return None
    h = hand[present]
    local = h - h[:, :1]                                  # wrist-centered
    scale = np.linalg.norm(local[:, 9], axis=-1).mean() or 1.0
    local = local / scale
    jitter = float(np.mean(np.linalg.norm(np.diff(local, axis=0), axis=-1)))
    return jitter, float(present.mean()), float(np.ptp(local[..., 2]))


EXAMPLES = ROOT / "data/references/example_videos.json"
ASL_LEX = ROOT / "data/external/asl_lex/signdata.csv"


def target_hands() -> dict[str, int]:
    """How many hands the 3D view must show = what the example video shows.
    The tracker under-counts signs whose hands press together (bed), so a
    sign also counts as two-handed when ASL-LEX codes it as a two-handed
    type AND both hands appear in the example in at least 10% of frames."""
    import csv
    import json
    import re
    two_handed = set()
    for row in csv.DictReader(open(ASL_LEX, encoding="utf-8", errors="replace")):
        if row.get("SignType.2.0", "").startswith(("Symmetrical", "Asymmetrical", "Dominance")):
            two_handed.add(re.sub(r"_\d+$", "", row["EntryID"].lower()))
    out = {}
    for w, c in json.loads(EXAMPLES.read_text()).items():
        n = c.get("n_hands", 1)
        if n == 1 and w in two_handed and (c.get("both_hands_share", 0) >= 0.1 or w in VERIFIED_TWO_HANDED):
            n = 2
        out[w] = n
    return out


# Two-handed signs whose hands stack or press together in the example video,
# so the tracker sees only one hand there. Checked by eye on the example
# frames (2026-11-22): bed (palms pressed at the cheek), cook (hand flips on
# the other palm), school (clap). `same` was also flagged by ASL-LEX but its
# example is the one-handed Y-hand version, so it is NOT listed.
VERIFIED_TWO_HANDED = {"bed", "cook", "school"}


def hands_used(canon: np.ndarray) -> tuple[int, float]:
    arr = canon.reshape(len(canon), 75, 3)
    l = np.any(arr[:, 33:54] != 0, axis=(1, 2))
    r = np.any(arr[:, 54:75] != 0, axis=(1, 2))
    any_hand = (l | r).sum()
    share = float((l & r).sum() / any_hand) if any_hand else 0.0
    return (2 if share >= 0.3 else 1), share


def main() -> None:
    refs = pickle.load(open(REFS, "rb"))
    v4 = pd.read_parquet(V4, columns=["video_id", "gloss", "source"])
    v4["vid"] = v4.video_id.astype(str)
    dec = pd.read_csv(DECISIONS)
    ok = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = v4[v4.vid.isin(ok) & v4.source.isin(SOURCE_BONUS.keys())]
    raw = load_raw_lookup()

    want = target_hands()
    for word in sorted(refs):
        current = refs[word].get("display_sequence_raw_canon")
        current = current if current is not None else refs[word].get("representative_sequence_raw_canon")
        need = want.get(word)
        if word in KEEP_AS_IS and (need is None or current is None or hands_used(np.asarray(current))[0] == need):
            continue  # first signs keep their 3D unless the hand count is wrong
        best = None
        for _, r in v4[v4.gloss == word].iterrows():
            flat = raw.get(r["vid"])
            if flat is None:
                continue
            canon = canonicalize_handedness(np.asarray(flat, dtype=np.float32), target="right")
            q = quality(canon)
            if q is None or q[2] < 0.05:                    # no usable depth
                continue
            jitter, visible, _ = q
            n, share = hands_used(canon)
            if need is not None and n != need and not (need == 2 and share >= 0.1):
                continue  # must show the same number of hands as the video
            # two-handed: the clip showing both hands most clearly comes first
            score = (round(share, 1) if need == 2 else 0, -SOURCE_RANK[r["source"]], -jitter)
            if best is None or score > best[0]:
                best = (score, canon, f"{r['source']}:{r['vid']}", jitter, visible)
        if best is None:
            print(f"{word:10s} no candidate, keeps current 3D")
            continue
        refs[word]["display_sequence_raw_canon"] = best[1]
        refs[word]["display_source"] = best[2]
        print(f"{word:10s} -> {best[2]:44s} jitter={best[3]:.3f} visible={best[4]:.2f}")

    pickle.dump(refs, open(REFS, "wb"))
    print(f"Saved -> {REFS}")


if __name__ == "__main__":
    main()
