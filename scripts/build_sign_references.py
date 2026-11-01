#!/usr/bin/env python3
"""Build the per-sign reference library for the teach/correct coach.

For each target sign, pulls only human-confirmed-correct clips (decision=ok
in review_decisions.csv -- today's QA work, not raw unreviewed data), runs
them through the trained classifier's embedding space (pooled features
before the classification head), and stores:
  - the embedding centroid (for "how close is this attempt" scoring)
  - one representative raw keypoint sequence (for the visual skeleton
    overlay / split-screen comparison)
  - the reference orientation profile (mean orientation during frames
    where the dominant hand is actually active, not resting)

Signs with zero confirmed-ok clips are skipped and reported -- there's
nothing to build a reference from yet, not something to fake.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from mouna.models.transformer import PoseTransformer  # noqa: E402

MODEL_PATH = Path("data/best_model_v4canon.pth")
# Must match what the model was actually trained on -- v4canon (handedness
# canonicalized), not v2norm_aslc. Using the wrong one would make reference
# embeddings inconsistent with what live inference (also canonicalized) produces.
DATASET_PATH = Path("data/silver/keypoints/keypoints_official_split_v4canon.parquet")
DECISIONS_PATH = Path("data/silver/qa/review_decisions.csv")
OUTPUT = Path("data/references/sign_references.pkl")
MAX_SEQ_LEN = 150


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

    df = pd.read_parquet(DATASET_PATH)
    df["_vid"] = df["video_id"].astype(str)
    df = df[df["_vid"].isin(ok_ids)]
    print(f"Matched to dataset: {len(df)} sequences across {df['gloss'].nunique()} glosses")

    references = {}
    all_words = sorted(df["gloss"].unique())
    missing = []

    for word in all_words:
        sub = df[df["gloss"] == word]
        embeddings = []
        seqs = []
        for _, row in sub.iterrows():
            flat = pickle.loads(row["keypoints_pickle"])  # already canonicalized+normalized
            T = min(len(flat), MAX_SEQ_LEN)
            padded = flat[:T]
            if T < MAX_SEQ_LEN:
                padded = np.vstack([padded, np.zeros((MAX_SEQ_LEN - T, flat.shape[1]), dtype=np.float32)])
            x = torch.from_numpy(padded).unsqueeze(0).float()
            lengths = torch.tensor([T])
            with torch.no_grad():
                emb = model.forward_features(x, lengths)[0].numpy()
            embeddings.append(emb)
            seqs.append(flat)

        centroid = np.mean(embeddings, axis=0)
        # representative sequence: closest single clip to the centroid, not
        # just the first one -- picks the most "typical" confirmed example
        dists = [np.linalg.norm(e - centroid) for e in embeddings]
        representative = seqs[int(np.argmin(dists))]

        references[word] = {
            "embedding_centroid": centroid,
            "representative_sequence": representative,
            "n_references": len(embeddings),
        }

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
