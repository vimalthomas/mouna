#!/usr/bin/env python3
"""Handshape as joint-angle vectors vs today's CP-profile handshape score,
on the same held-out attempts and the same ASL-LEX answer key.

Fluent clouds: each sign's approved clips (main form). Attempts: the judgement
benchmark's held-out clips that have a video (needed for true proportions).
Reports AUC (attempts whose handshape truly differs vs correct signing) for
both methods, overall and on minimal pairs (only the handshape differs).
Writes data/references/judgement_benchmark/vector_handshape.json
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
USE_DEPTH = os.environ.get("VEC_DEPTH", "1") == "1"
OWN = float(os.environ.get("VEC_OWN", "0"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.cp_profile import features  # noqa: E402
from mouna.coach.scores import score_parameters  # noqa: E402
from mouna.coach.vectors import Cloud, HS_NAMES, handshape_vector  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    lex = {}
    for r in csv.DictReader(open(ROOT / "data/external/asl_lex/signdata.csv", encoding="utf-8", errors="replace")):
        lex.setdefault(re.sub(r"_\d+$", "", r["EntryID"].lower()), r)
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
        vs = [handshape_vector(canon(v), *sizes[v], use_depth=USE_DEPTH) for v in sub.vid if v not in drop and v in raw and v in sizes]
        vs = [x for x in vs if x is not None]
        if len(vs) >= 5:
            vecs[sign] = vs
    allv = np.concatenate([np.stack(v) for v in vecs.values()])
    scale = np.maximum(np.std(allv, axis=0), 1.0)
    clouds = {s: Cloud(v, scale, own_weight=OWN) for s, v in vecs.items()}
    print(f"fluent clouds: {len(clouds)} signs, {len(allv)} clips (median {int(np.median([len(v) for v in vecs.values()]))} per sign)")

    rows = pd.read_csv(ROOT / "data/references/judgement_benchmark/rows.csv.gz")
    rows["vid"] = rows.video_id.astype(str)
    rows = rows[rows.vid.isin(sizes) & rows.vid.isin(raw) & rows.target.isin(clouds) & rows.target.isin(prof)]
    def val(r, c):
        x = (r.get(c) or "").strip(); return None if not x or x == "NA" else x
    out = []
    cache = {}
    for r in rows.itertuples():
        a, b = lex.get(r.target), lex.get(r.attempt_sign)
        if not a or not b:
            continue
        if r.vid not in cache:
            c = canon(r.vid)
            cache[r.vid] = (handshape_vector(c, *sizes[r.vid], use_depth=USE_DEPTH), features(c))
        v, f = cache[r.vid]
        if v is None or f is None:
            continue
        hd = val(a, "Handshape.2.0") != val(b, "Handshape.2.0") if val(a, "Handshape.2.0") and val(b, "Handshape.2.0") else None
        others_same = all(val(a, c) == val(b, c) for c in ("MinorLocation.2.0", "Movement.2.0", "SignType.2.0") if val(a, c) and val(b, c))
        out.append({"target": r.target, "same": bool(r.same), "hs_differs": hd, "minimal": bool(hd and others_same),
                    "vector": clouds[r.target].unusualness(v),
                    "ratio": clouds[r.target].distance(v) / max(float(np.median(clouds[r.target].null)), 1e-9),
                    "old": score_parameters(f, prof[r.target], cal={})["handshape"]["deviation"]})
    d = pd.DataFrame(out)
    # both methods together: average of each one's rank (they measure different things)
    d["combined"] = (d.vector.rank(pct=True) + d.old.rank(pct=True)) / 2
    d["combined_ratio"] = (d.ratio.rank(pct=True) + d.old.rank(pct=True)) / 2

    def auc(mask_diff, col):
        pos, neg = d.loc[mask_diff, col].dropna(), d.loc[d.same, col].dropna()
        if not len(pos) or not len(neg):
            return None
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        return round(float(roc_auc_score(y, np.r_[neg, pos])), 3)
    def catches_at_10(mask, col):
        cut = d.loc[d.same, col].dropna().quantile(0.9)
        x = d.loc[mask, col].dropna()
        return round(float((x > cut).mean()), 3) if len(x) else None
    res = {"correct": int(d.same.sum()), "handshape_differs": int(d.hs_differs.eq(True).sum()), "minimal_pairs": int(d.minimal.sum()),
           "auc_all": {c: auc(d.hs_differs.eq(True), c) for c in ("vector", "ratio", "old", "combined", "combined_ratio")},
           "auc_minimal_pairs": {c: auc(d.minimal, c) for c in ("vector", "ratio", "old", "combined", "combined_ratio")},
           "catches_at_10pct_false_alarm": {c: catches_at_10(d.hs_differs.eq(True), c) for c in ("vector", "ratio", "old", "combined", "combined_ratio")},
           "correct_typical": round(float((d.loc[d.same, "vector"] <= 0.9).mean()), 3)}
    res["settings"] = {"depth": USE_DEPTH, "own_weight": OWN}
    d.rename(columns={"hs_differs": "differs"}).to_csv(ROOT / "data/references/judgement_benchmark/vector_handshape_rows.csv.gz", index=False)
    (ROOT / f"data/references/judgement_benchmark/vector_handshape{'' if USE_DEPTH and not OWN else '_d%d_w%s' % (USE_DEPTH, OWN)}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
