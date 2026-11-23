#!/usr/bin/env python3
"""Can the recognizer's embedding grade signs it was never trained on?

The verdict is cosine similarity of an attempt to the sign's reference
centroid (good >= GOOD_SIMILARITY_THRESHOLD). For each sign, centroids are
built from half the signers and tested on the other half (no signer in both),
for the 8 new signs and, as the yardstick, the trained signs. Reported:
  hit    - held-out clips of the sign that score "good" against it
  false  - clips of OTHER signs that would also score "good" against it
  top1   - held-out clips whose nearest centroid (all signs) is the right one
Run: ./venv/bin/python scripts/validate_untrained_signs.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path[:0] = ["src", "scripts"]
from build_sign_references_v2 import DECISIONS_PATH, MAX_SEQ_LEN, MODEL_PATH, V4CANON_PATH, load_raw_lookup  # noqa: E402
from mouna.coach.compare import GOOD_SIMILARITY_THRESHOLD as GOOD  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.data.part_normalize import normalize_per_part  # noqa: E402
from mouna.models.transformer import PoseTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data/silver/keypoints_v2/keypoints_aslc_new_raw.parquet"


def main():
    ck = torch.load(ROOT / MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = PoseTransformer(**{k: v for k, v in cfg.items() if k != "num_classes"}, num_classes=cfg["num_classes"])
    m.load_state_dict(ck["model_state_dict"]); m.eval()

    def emb(flat):
        n = normalize_per_part(canonicalize_handedness(np.asarray(flat, dtype=np.float32), target="right"))
        T = min(len(n), MAX_SEQ_LEN); x = np.zeros((MAX_SEQ_LEN, n.shape[1]), np.float32); x[:T] = n[:T]
        with torch.no_grad():
            e = m.forward_features(torch.from_numpy(x)[None], torch.tensor([T]))[0].numpy()
        return e / (np.linalg.norm(e) + 1e-8)

    rows = []
    qa = pd.read_csv(ROOT / "data/silver/qa/new_aslc_clips_qa.csv").set_index("video_file")
    for _, r in pd.read_parquet(NEW).query("success").iterrows():
        rows.append((r.gloss, qa.participant_id.get(r.video_id, r.video_id), emb(pickle.loads(r.keypoints_pickle)), True))
    dec = pd.read_csv(ROOT / DECISIONS_PATH); ok = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / V4CANON_PATH, columns=["video_id", "gloss", "source"])
    v4["vid"] = v4.video_id.astype(str); v4 = v4[v4.vid.isin(ok) & (v4.source != "openpose_recovered")]
    raw = load_raw_lookup()
    for _, r in v4.iterrows():
        f = raw.get(r.vid)
        if f is not None:
            rows.append((r.gloss, r.vid.split("-")[0] if r.source == "asl_citizen" else r.vid, emb(f), False))
    df = pd.DataFrame(rows, columns=["sign", "signer", "e", "new"])
    df = df[df.groupby("sign").sign.transform("size") >= 6]

    rng = np.random.default_rng(0)
    train, test = [], []
    for s, g in df.groupby("sign"):
        signers = g.signer.unique(); rng.shuffle(signers)
        half = set(signers[: len(signers) // 2])
        train.append(g[g.signer.isin(half)]); test.append(g[~g.signer.isin(half)])
    train, test = pd.concat(train), pd.concat(test)
    signs = sorted(train.sign.unique())
    C = np.stack([np.mean(np.stack(train[train.sign == s].e), 0) for s in signs])
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    E = np.stack(test.e); sims = E @ C.T
    lab = np.array([signs.index(s) for s in test.sign])
    out = []
    for i, s in enumerate(signs):
        mine = lab == i
        out.append({"sign": s, "new": bool(test[mine].new.iloc[0]), "n_test": int(mine.sum()),
                    "hit": (sims[mine, i] >= GOOD).mean(), "false": (sims[~mine, i] >= GOOD).mean(),
                    "top1": (sims[mine].argmax(1) == i).mean(), "sim": sims[mine, i].mean()})
    r = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(r[r.new].round(2).to_string(index=False))
    print("\nmean   new signs:", r[r.new][["hit", "false", "top1", "sim"]].mean().round(3).to_dict())
    print("mean   trained  :", r[~r.new][["hit", "false", "top1", "sim"]].mean().round(3).to_dict())
    r.to_csv(ROOT / "data/references/untrained_sign_validation.csv", index=False)


if __name__ == "__main__":
    main()
