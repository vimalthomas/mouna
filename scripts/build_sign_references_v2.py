#!/usr/bin/env python3
"""Rebuild the per-sign reference library with an ADDITIONAL raw-canonicalized
(pre-normalize_per_part) representative sequence per word, needed for the new
decomposed parameter scoring in src/mouna/coach/parameters.py.

Why this exists instead of just reusing build_sign_references.py: that script
reads keypoints_official_split_v4canon.parquet, where keypoints_pickle is
ALREADY canonicalize_handedness + normalize_per_part'd -- the intermediate
canonicalized-but-not-yet-part-normalized form isn't persisted there. Location
scoring needs exactly that intermediate form (hand position relative to the
body frame -- normalize_per_part's per-part centering throws this away by
design, see part_normalize.py's docstring). So this script re-derives it from
the same raw sources build_official_split_v4canon.py reads, applies
canonicalize_handedness only, and keeps both forms side by side:
  - representative_sequence              (canon + normalized) -- unchanged,
    still what the model/embedding centroid uses.
  - representative_sequence_raw_canon    (canon only) -- new, for
    parameters.py's location/handshape/orientation geometry.

Also fixes a latent inconsistency: compare.py's hand_count_match calls
detect_dominant_hand() on representative_sequence, but that function's own
docstring says it expects RAW (pre-normalize) keypoints -- it was being run
against the normalized form. This script's raw_canon field is what that call
should actually use (wired up in compare.py alongside this).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.data.openpose_convert import convert_video as op_convert_video  # noqa: E402
from mouna.data.part_normalize import normalize_per_part  # noqa: E402
from mouna.coach.parameters import extract_dominant_hand_signal, majority_targets  # noqa: E402
from mouna.models.transformer import PoseTransformer  # noqa: E402

MODEL_PATH = Path("data/best_model_v4canon.pth")
V4CANON_PATH = Path("data/silver/keypoints/keypoints_official_split_v4canon.parquet")
DECISIONS_PATH = Path("data/silver/qa/review_decisions.csv")
RAW_MEDIAPIPE = Path("data/silver/keypoints/keypoints_raw.parquet")
RAW_MSASL = Path("data/silver/keypoints/keypoints_msasl_raw.parquet")
RAW_ASLC = Path("data/silver/keypoints/keypoints_asl_citizen_raw.parquet")
OUTPUT = Path("data/references/sign_references.pkl")
MAX_SEQ_LEN = 150


def load_raw_lookup():
    """video_id (str) -> raw [0,1]-space flat keypoints, dispatched by source,
    mirroring build_official_split_v4canon.py's own source dispatch exactly."""
    lookup = {}

    raw_mp = pd.read_parquet(RAW_MEDIAPIPE)
    raw_mp = raw_mp[raw_mp["success"]]
    for _, row in raw_mp.iterrows():
        lookup[str(row["video_id"])] = pickle.loads(row["keypoints_pickle"])

    raw_msasl = pd.read_parquet(RAW_MSASL)
    raw_msasl = raw_msasl[raw_msasl["success"]]
    for _, row in raw_msasl.iterrows():
        lookup[str(row["video_id"])] = pickle.loads(row["keypoints_pickle"])

    raw_aslc = pd.read_parquet(RAW_ASLC)
    raw_aslc = raw_aslc[raw_aslc["success"]]
    for _, row in raw_aslc.iterrows():
        lookup[str(row["video_id"])] = pickle.loads(row["keypoints_pickle"])

    return lookup


def main():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = PoseTransformer(**{k: v for k, v in cfg.items() if k != "num_classes"}, num_classes=cfg["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {cfg['num_classes']} classes")

    decisions = pd.read_csv(DECISIONS_PATH)
    ok_ids = set(decisions[decisions["decision"] == "ok"]["video_id"].astype(str))
    print(f"Confirmed-ok clips available: {len(ok_ids)}")

    v4 = pd.read_parquet(V4CANON_PATH)
    v4["_vid"] = v4["video_id"].astype(str)
    v4 = v4[v4["_vid"].isin(ok_ids)]
    print(f"Matched to dataset: {len(v4)} sequences across {v4['gloss'].nunique()} glosses")

    raw_lookup = load_raw_lookup()
    print(f"Raw-source lookup built: {len(raw_lookup)} entries")

    references = {}
    all_words = sorted(v4["gloss"].unique())

    for word in all_words:
        sub = v4[v4["gloss"] == word]
        embeddings, normed_seqs, raw_canon_seqs = [], [], []

        for _, row in sub.iterrows():
            vid = row["_vid"]
            source = row["source"]

            if source == "openpose_recovered":
                raw_flat = op_convert_video(vid)
            else:
                raw_flat = raw_lookup.get(vid)
                if raw_flat is None:
                    continue  # shouldn't happen -- every non-openpose row came from one of the 3 raw parquets

            canon = canonicalize_handedness(raw_flat, target="right")
            normed = normalize_per_part(canon)

            T = min(len(normed), MAX_SEQ_LEN)
            padded = normed[:T]
            if T < MAX_SEQ_LEN:
                padded = np.vstack([padded, np.zeros((MAX_SEQ_LEN - T, padded.shape[1]), dtype=np.float32)])
            x = torch.from_numpy(padded).unsqueeze(0).float()
            lengths = torch.tensor([T])
            with torch.no_grad():
                emb = model.forward_features(x, lengths)[0].numpy()

            embeddings.append(emb)
            normed_seqs.append(normed)
            raw_canon_seqs.append(canon)

        if not embeddings:
            print(f"  {word}: 0 clips resolved, skipping")
            continue

        centroid = np.mean(embeddings, axis=0)
        dists = [np.linalg.norm(e - centroid) for e in embeddings]
        best = int(np.argmin(dists))

        references[word] = {
            "embedding_centroid": centroid,
            "representative_sequence": normed_seqs[best],
            "representative_sequence_raw_canon": raw_canon_seqs[best],
            "n_references": len(embeddings),
            # CP targets by majority vote over every approved clip, not the
            # single representative one (see parameters.majority_targets).
            "target_buckets": majority_targets([extract_dominant_hand_signal(c) for c in raw_canon_seqs]),
        }
        print(f"  {word}: {len(embeddings)} clips")

    covered = set(references.keys())
    SIGNS_JSON = Path("app/backend/signs.json")
    full_vocab = {s["word"] for s in json.load(open(SIGNS_JSON))["signs"]}
    missing = sorted(full_vocab - covered)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "wb") as f:
        pickle.dump(references, f)

    print(f"\nReferences built for {len(references)}/{len(full_vocab)} vocabulary words")
    if missing:
        print(f"No confirmed-ok clips yet (skipped, not faked): {missing}")
    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
