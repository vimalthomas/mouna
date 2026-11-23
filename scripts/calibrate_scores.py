#!/usr/bin/env python3
"""Calibrate and validate the five parameter scores (mouna.coach.scores).

For each parameter, on held-out benchmark attempts:
  - correct attempts at the sign should score high
  - attempts of another sign that differs in THAT parameter (per ASL-LEX)
    should score low
AUC of the parameter's deviation for "differs" vs "correct" says how well the
score sees that parameter. The "needs work" cut-off is chosen on a TUNE half
of the signs (most real differences caught while flagging at most 10% of
correct signing) and every number is reported on the other, TEST half.
Palm orientation has no ASL-LEX coding: only its false alarms are measured.
Writes data/references/score_calibration.json.
"""
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.cp_profile import features  # noqa: E402
from mouna.coach.scores import PARAMETERS, score_parameters  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402

TRUTH = {"handshape": ("Handshape.2.0",), "location": ("MinorLocation.2.0",),
         "movement": ("Movement.2.0", "RepeatedMovement.2.0", "Contact.2.0"), "orientation": None}
MAX_FALSE_ALARM, MIN_AUC = 0.10, 0.75
GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
FLUENT_SHARE = 0.80   # "fluent" level: where 80% of correct signers fall (TUNE half)


def main() -> None:
    os.chdir(ROOT)
    lex = {}
    for r in csv.DictReader(open(ROOT / "data/external/asl_lex/signdata.csv", encoding="utf-8", errors="replace")):
        lex.setdefault(re.sub(r"_\d+$", "", r["EntryID"].lower()), r)
    prof = json.loads((ROOT / "data/references/cp_profiles.json").read_text())
    rows = pd.read_csv(ROOT / "data/references/judgement_benchmark/rows.csv.gz")
    raw = load_raw_lookup()
    F = {}
    def feats(v):
        if v not in F:
            F[v] = features(canonicalize_handedness(np.asarray(raw[v], dtype=np.float32), target="right")) if v in raw else None
        return F[v]
    def val(r, c):
        x = (r.get(c) or "").strip(); return None if not x or x == "NA" else x

    items = []
    for r in rows.itertuples():
        f, p, a, b = feats(str(r.video_id)), prof.get(r.target), lex.get(r.target), lex.get(r.attempt_sign)
        if f is None or not p or not a or not b:
            continue
        sc = score_parameters(f, p, cal={})
        diff = {}
        for par, cols in TRUTH.items():
            if cols is None:
                diff[par] = None
                continue
            pairs = [(val(a, c), val(b, c)) for c in cols]
            diff[par] = None if any(x is None or y is None for x, y in pairs) else any(x != y for x, y in pairs)
        items.append((r.target, bool(r.same), {k: v["deviation"] for k, v in sc.items()}, diff))

    signs = sorted({t for t, *_ in items}); random.Random(9).shuffle(signs)
    tune = set(signs[: len(signs) // 2])
    out = {"max_false_alarm": MAX_FALSE_ALARM, "min_auc": MIN_AUC, "tune_signs": sorted(tune), "parameters": {}}
    for par in PARAMETERS:
        if par == "nonmanual":
            out["parameters"][par] = {"measured": False}
            continue
        def collect(subset):
            cor = [d[par] for t, s, d, _ in items if s and d[par] is not None and ((t in tune) == (subset == "tune"))]
            dif = [d[par] for t, s, d, df in items if not s and df[par] and d[par] is not None and ((t in tune) == (subset == "tune"))]
            return np.array(cor), np.array(dif)
        ct, dt = collect("tune")
        # cut-off: needs_work = deviation > cutoff; at most 10% of correct signing flagged
        ok = [c for c in GRID if len(ct) and (ct > c).mean() <= MAX_FALSE_ALARM]
        cut = min(ok) if ok else GRID[-1]
        fluent_cut = float(np.quantile(ct, FLUENT_SHARE)) if len(ct) else 0.0
        ce, de = collect("test")
        info = {"cutoff": cut, "fluent_cutoff": round(min(fluent_cut, cut), 3), "test_false_alarm": round(float((ce > cut).mean()), 3) if len(ce) else None,
                "test_correct_fluent": round(float((ce <= min(fluent_cut, cut)).mean()), 3) if len(ce) else None, "n_correct": int(len(ce))}
        if len(de) and len(ce):
            y = np.r_[np.zeros(len(ce)), np.ones(len(de))]
            info["test_auc"] = round(float(roc_auc_score(y, np.r_[ce, de])), 3)
            info["test_catches"] = round(float((de > cut).mean()), 3)
            info["n_differs"] = int(len(de))
        info["reliable"] = bool(info.get("test_auc") and info["test_auc"] >= MIN_AUC and info["test_false_alarm"] <= MAX_FALSE_ALARM)
        out["parameters"][par] = info
        print(f"{par:12s} cutoff {cut:<4}  TEST: AUC {info.get('test_auc')}  catches {info.get('test_catches')}  "
              f"false alarm {info['test_false_alarm']}  correct scored fluent {info['test_correct_fluent']}  reliable {info['reliable']}")
    (ROOT / "data/references/score_calibration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
