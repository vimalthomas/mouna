#!/usr/bin/env python3
"""Old (v4canon, 100 signs) vs new (v5new8, 108 signs) on the same held-out
test clips: top-1 on the old WLASL test split, and on the new signs' unseen
ASL Citizen signers. Usage: compare_models.py [old.pth] [new.pth]"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mouna.models.transformer import PoseTransformer  # noqa: E402

DATA = ROOT / "data/silver/keypoints_v2/keypoints_official_split_v5new8.parquet"


def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False)
    m = PoseTransformer(**c["config"]); m.load_state_dict(c["model_state_dict"]); m.eval()
    return m, c["idx_to_label"]


def acc(model, labels, df):
    hits = []
    for _, r in df.iterrows():
        x = pickle.loads(r.keypoints_pickle)[:150]
        T = len(x); pad = np.zeros((150, x.shape[1]), np.float32); pad[:T] = x
        with torch.no_grad():
            p = model(torch.from_numpy(pad)[None], torch.tensor([T])).argmax(-1).item()
        hits.append(labels[p] == r.gloss)
    return float(np.mean(hits)), len(hits)


def main():
    old_p = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/best_model_v4canon.pth"
    new_p = sys.argv[2] if len(sys.argv) > 2 else ROOT / "data/best_model_v5new8.pth"
    df = pd.read_parquet(DATA); test = df[df.official_split == "test"]
    old_t, new_t = test[test.source != "asl_citizen_new8"], test[test.source == "asl_citizen_new8"]
    (mo, lo), (mn, ln) = load(old_p), load(new_p)
    print("old-sign test (WLASL split): v4canon %.3f  v5new8 %.3f  (n=%d)" % (acc(mo, lo, old_t)[0], acc(mn, ln, old_t)[0], len(old_t)))
    print("new-sign test (unseen signers): v5new8 %.3f  (n=%d; v4canon can't name them)" % (acc(mn, ln, new_t)[0], len(new_t)))


if __name__ == "__main__":
    main()
