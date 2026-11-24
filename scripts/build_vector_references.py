#!/usr/bin/env python3
"""Fluent-signer clouds + calibration for the live five-parameter scores
(mouna.coach.param_scores). Writes data/references/vector_references.pkl.

Per sign: handshape joint-angle cloud (3D, own-weight 0.5), location cloud,
movement path cloud (DTW) -- built from approved main-form clips, exactly as
in vector_benchmark.py / param_vector_benchmark.py / movement_vector_benchmark.py.

Calibration (per parameter), from those benchmarks' per-attempt rows and
using ONLY the tune signs (cp_tolerances.json):
  combined = average of the attempt's percentile among all tune-sign
             attempts (correct and wrong) on the vector ratio and on the CP
             deviation (as in vector_split_check.py; ranking among correct
             attempts only was also tried: movement then caught 41% on test)
  fluent cut     = 80th percentile of correct attempts' combined
  needs-work cut = 90th percentile (flags at most 10% of correct signing)
The same formula, with these cuts, is then measured on the test signs; a
parameter is "reliable" only if there false alarm <= 10% and catches >= 50%.
Run the three benchmarks first (they write vector_*_rows.csv.gz).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.vectors import Cloud, PathCloud, handshape_vector, location_vector, movement_path  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402

B = ROOT / "data/references/judgement_benchmark"
SPLIT = json.loads((ROOT / "data/references/cp_tolerances.json").read_text())
TUNE, TEST = set(SPLIT["tune_signs"]), set(SPLIT["test_signs"])


def pct(sorted_arr: np.ndarray, x) -> np.ndarray:
    """Mid-rank percentile of x among the sorted values."""
    return (np.searchsorted(sorted_arr, x, "left") + np.searchsorted(sorted_arr, x, "right")) / 2 / len(sorted_arr)


def calibrate(param: str) -> dict:
    d = pd.read_csv(B / f"vector_{param}_rows.csv.gz")
    d["differs"] = d.differs.fillna(False).astype(bool)
    d["old"] = d.old.fillna(0.0)
    tune, test = d[d.target.isin(TUNE)], d[d.target.isin(TEST)]
    tc = tune[tune.same]
    ratio_c, old_c = np.sort(tune.ratio.values), np.sort(tune.old.values)
    comb = lambda df: (pct(ratio_c, df.ratio.values) + pct(old_c, df.old.values)) / 2  # noqa: E731
    fc, nc = float(np.quantile(comb(tc), 0.8)), float(np.quantile(comb(tc), 0.9))
    te_c, te_d = test[test.same], test[test.differs & ~test.same]
    fa, ca = float((comb(te_c) > nc).mean()), float((comb(te_d) > nc).mean())
    fluent_ok = float((comb(te_c) <= fc).mean())
    return {"ratio_ranks": ratio_c, "cp_ranks": old_c, "fluent_cut": fc, "needs_work_cut": nc,
            "test_false_alarm": round(fa, 3), "test_catches": round(ca, 3), "test_correct_fluent": round(fluent_ok, 3),
            "n_tune_correct": len(tc), "n_test_correct": len(te_c), "n_test_differs": len(te_d),
            "reliable": fa <= 0.10 and ca >= 0.50}


def main() -> None:
    sizes = json.loads((ROOT / "data/references/video_sizes.json").read_text())
    raw = load_raw_lookup()
    canon = lambda v: canonicalize_handedness(np.asarray(raw[v], dtype=np.float32), target="right")  # noqa: E731
    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    ok = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss"])
    v4["vid"] = v4.video_id.astype(str)
    variants = json.loads((ROOT / "data/references/sign_variants.json").read_text())
    per = {"handshape": {}, "location": {}, "movement": {}}
    for sign, sub in v4[v4.vid.isin(ok)].groupby("gloss"):
        drop = {i for alt in variants.get(sign, {}).get("variations", []) for i in alt["video_ids"]}
        vids = [v for v in sub.vid if v not in drop and v in raw and v in sizes]
        seqs = [(canon(v), sizes[v]) for v in vids]
        for name, fn in (("handshape", handshape_vector), ("location", location_vector), ("movement", movement_path)):
            xs = [x for x in (fn(c, *wh) for c, wh in seqs) if x is not None]
            if len(xs) >= 5:
                per[name][sign] = xs
    clouds = {}
    for name, floor, own in (("handshape", 1.0, 0.5), ("location", 0.05, 0.0)):
        allv = np.concatenate([np.stack(v) for v in per[name].values()])
        scale = np.maximum(np.std(allv, axis=0), floor)
        clouds[name] = {s: Cloud(v, scale, own_weight=own) for s, v in per[name].items()}
    clouds["movement"] = {s: PathCloud(v) for s, v in per["movement"].items()}
    cal = {p: calibrate(p) for p in ("handshape", "location", "movement")}
    out = {"clouds": clouds, "calibration": cal, "version": 1}
    with open(ROOT / "data/references/vector_references.pkl", "wb") as f:
        pickle.dump(out, f)
    for p, c in cal.items():
        print(f"{p:10s} signs={len(clouds[p]):3d}  test: false alarm {c['test_false_alarm']:.3f}  catches {c['test_catches']:.3f}  "
              f"correct fluent {c['test_correct_fluent']:.3f}  reliable={c['reliable']}")


if __name__ == "__main__":
    main()
