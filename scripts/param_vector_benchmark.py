#!/usr/bin/env python3
"""Palm orientation and location as vectors vs today's CP-profile scores, on
the judgement benchmark's held-out attempts (same method as vector_benchmark.py).

  VEC_PARAM=location   answer key: ASL-LEX location differs (major/minor);
                       minimal pairs: only the location differs
  VEC_PARAM=palm       ASL-LEX does not code palm orientation. Stand-in:
                       wrong signs that ASL-LEX codes as identical in
                       handshape, location, movement and hand count -- what
                       separates them is uncoded (often orientation)
VEC_OWN blends each sign's own spread into the scale (0 = off).
Writes data/references/judgement_benchmark/vector_<param>[_w<own>].json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PARAM = os.environ.get("VEC_PARAM", "location")
OWN = float(os.environ.get("VEC_OWN", "0"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.cp_profile import features  # noqa: E402
from mouna.coach.scores import score_parameters  # noqa: E402
from mouna.coach.vectors import Cloud, location_vector, palm_vector  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402

VEC = {"location": location_vector, "palm": palm_vector}[PARAM]
OLD = {"location": "location", "palm": "orientation"}[PARAM]


def main() -> None:
    os.chdir(ROOT)
    sizes = json.loads((ROOT / "data/references/video_sizes.json").read_text())
    prof = json.loads((ROOT / "data/references/cp_profiles.json").read_text())
    raw = load_raw_lookup()
    canon = lambda v: canonicalize_handedness(np.asarray(raw[v], dtype=np.float32), target="right")  # noqa: E731

    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    ok = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss"])
    v4["vid"] = v4.video_id.astype(str)
    variants = json.loads((ROOT / "data/references/sign_variants.json").read_text())
    vecs = {}
    for sign, sub in v4[v4.vid.isin(ok)].groupby("gloss"):
        drop = {i for alt in variants.get(sign, {}).get("variations", []) for i in alt["video_ids"]}
        vs = [VEC(canon(v), *sizes[v]) for v in sub.vid if v not in drop and v in raw and v in sizes]
        vs = [x for x in vs if x is not None]
        if len(vs) >= 5:
            vecs[sign] = vs
    allv = np.concatenate([np.stack(v) for v in vecs.values()])
    scale = np.maximum(np.std(allv, axis=0), 0.05)
    clouds = {s: Cloud(v, scale, own_weight=OWN) for s, v in vecs.items()}
    print(f"{PARAM}: fluent clouds {len(clouds)} signs, {len(allv)} clips")

    rows = pd.read_csv(ROOT / "data/references/judgement_benchmark/rows.csv.gz")
    rows["vid"] = rows.video_id.astype(str)
    rows = rows[rows.vid.isin(sizes) & rows.vid.isin(raw) & rows.target.isin(clouds) & rows.target.isin(prof)]
    out, cache = [], {}
    for r in rows.itertuples():
        if r.vid not in cache:
            c = canon(r.vid)
            cache[r.vid] = (VEC(c, *sizes[r.vid]), features(c))
        v, f = cache[r.vid]
        if v is None or f is None:
            continue
        hs, loc, mov, hands = bool(r.differs_handshape), bool(r.differs_location or r.differs_location_detail), bool(r.differs_movement), bool(r.differs_hands)
        if PARAM == "location":
            differs, minimal = loc, loc and not (hs or mov or hands)
        else:
            differs = minimal = (not r.same) and not (hs or loc or mov or hands)
        old = score_parameters(f, prof[r.target], cal={})[OLD]["deviation"]
        out.append({"target": r.target, "same": bool(r.same), "differs": differs, "minimal": minimal,
                    "vector": clouds[r.target].unusualness(v),
                    "ratio": clouds[r.target].distance(v) / max(float(np.median(clouds[r.target].null)), 1e-9), "old": np.nan if old is None else old})
    d = pd.DataFrame(out)
    d["combined"] = (d.vector.rank(pct=True) + d.old.rank(pct=True)) / 2
    d["combined_ratio"] = (d.ratio.rank(pct=True) + d.old.rank(pct=True)) / 2

    def auc(mask, col):
        pos, neg = d.loc[mask, col].dropna(), d.loc[d.same, col].dropna()
        if not len(pos) or not len(neg):
            return None
        return round(float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])), 3)
    def catches_at_10(mask, col):
        cut = d.loc[d.same, col].dropna().quantile(0.9)     # flags at most 10% of correct attempts
        x = d.loc[mask, col].dropna()
        return round(float((x > cut).mean()), 3) if len(x) else None
    res = {"param": PARAM, "correct": int(d.same.sum()), "differs": int(d.differs.sum()), "minimal_pairs": int(d.minimal.sum()),
           "auc_all": {c: auc(d.differs, c) for c in ("vector", "ratio", "old", "combined", "combined_ratio")},
           "auc_minimal_pairs": {c: auc(d.minimal, c) for c in ("vector", "ratio", "old", "combined", "combined_ratio")},
           "catches_at_10pct_false_alarm": {c: catches_at_10(d.differs, c) for c in ("vector", "ratio", "old", "combined", "combined_ratio")},
           "correct_typical": round(float((d.loc[d.same, "vector"] <= 0.9).mean()), 3), "own_weight": OWN}
    d.to_csv(ROOT / f"data/references/judgement_benchmark/vector_{PARAM}_rows.csv.gz", index=False)
    (ROOT / f"data/references/judgement_benchmark/vector_{PARAM}{'' if not OWN else '_w%s' % OWN}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
