#!/usr/bin/env python3
"""Movement as a time-aligned path (DTW against the fluent signers of the sign)
vs today's CP-profile movement score, on the judgement benchmark's held-out
attempts. Answer key: ASL-LEX movement differs; minimal pairs: only the
movement differs. Writes vector_movement.json and vector_movement_rows.csv.gz
(for vector_split_check.py).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.cp_profile import features  # noqa: E402
from mouna.coach.scores import score_parameters  # noqa: E402
from mouna.coach.vectors import PathCloud, movement_path  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    t0 = time.time()
    sizes = json.loads((ROOT / "data/references/video_sizes.json").read_text())
    prof = json.loads((ROOT / "data/references/cp_profiles.json").read_text())
    raw = load_raw_lookup()
    canon = lambda v: canonicalize_handedness(np.asarray(raw[v], dtype=np.float32), target="right")  # noqa: E731
    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    ok = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss"])
    v4["vid"] = v4.video_id.astype(str)
    variants = json.loads((ROOT / "data/references/sign_variants.json").read_text())
    clouds = {}
    for sign, sub in v4[v4.vid.isin(ok)].groupby("gloss"):
        drop = {i for alt in variants.get(sign, {}).get("variations", []) for i in alt["video_ids"]}
        ps = [movement_path(canon(v), *sizes[v]) for v in sub.vid if v not in drop and v in raw and v in sizes]
        ps = [p for p in ps if p is not None]
        if len(ps) >= 5:
            clouds[sign] = PathCloud(ps)
    print(f"movement: fluent clouds {len(clouds)} signs ({time.time() - t0:.0f}s)")

    rows = pd.read_csv(ROOT / "data/references/judgement_benchmark/rows.csv.gz")
    rows["vid"] = rows.video_id.astype(str)
    rows = rows[rows.vid.isin(sizes) & rows.vid.isin(raw) & rows.target.isin(clouds) & rows.target.isin(prof)]
    cache = {}
    for vid in rows.vid.unique():
        c = canon(vid)
        cache[vid] = (movement_path(c, *sizes[vid]), features(c))
    out = []
    for target, g in rows.groupby("target"):
        g = g[[cache[v][0] is not None and cache[v][1] is not None for v in g.vid]]
        if not len(g):
            continue
        ratio = clouds[target].ratio(np.stack([cache[v][0] for v in g.vid]))
        for r, q in zip(g.itertuples(), ratio):
            mv, hs = bool(r.differs_movement), bool(r.differs_handshape)
            loc, hands = bool(r.differs_location or r.differs_location_detail), bool(r.differs_hands)
            old = score_parameters(cache[r.vid][1], prof[target], cal={})["movement"]["deviation"]
            out.append({"target": target, "same": bool(r.same), "differs": mv, "minimal": mv and not (hs or loc or hands),
                        "ratio": float(q), "old": np.nan if old is None else old})
    d = pd.DataFrame(out)
    d["combined_ratio"] = (d.ratio.rank(pct=True) + d.old.rank(pct=True)) / 2
    print(f"scored {len(d)} attempts ({time.time() - t0:.0f}s)")

    def auc(mask, col):
        pos, neg = d.loc[mask & ~d.same, col].dropna(), d.loc[d.same, col].dropna()
        if not len(pos) or not len(neg):
            return None
        return round(float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])), 3)
    cols = ("ratio", "old", "combined_ratio")
    res = {"param": "movement", "correct": int(d.same.sum()), "differs": int((d.differs & ~d.same).sum()), "minimal_pairs": int(d.minimal.sum()),
           "auc_all": {c: auc(d.differs, c) for c in cols}, "auc_minimal_pairs": {c: auc(d.minimal, c) for c in cols}}
    d.to_csv(ROOT / "data/references/judgement_benchmark/vector_movement_rows.csv.gz", index=False)
    (ROOT / "data/references/judgement_benchmark/vector_movement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
