#!/usr/bin/env python3
"""Held-out check of the vector scores as yes/no checks: the cut that flags
at most 10% of correct attempts is chosen on the tune signs and applied to
the test signs (same split as calibrate_cp_checks.py). A check is reliable if
on the test signs false alarm <= 10% and catches >= 50%.
Reads vector_<param>_rows.csv.gz written by vector_benchmark.py /
param_vector_benchmark.py; writes vector_split_check.json.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "data/references/judgement_benchmark"
split = json.loads((ROOT / "data/references/cp_tolerances.json").read_text())
TUNE, TEST = set(split["tune_signs"]), set(split["test_signs"])
out = {}
for param in ("handshape", "location", "movement", "palm"):
    f = B / f"vector_{param}_rows.csv.gz"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    d["differs"] = d.differs.fillna(False).astype(bool)
    res = {}
    for col in ("ratio", "old", "combined_ratio"):
        if param == "palm":        # too few stand-in pairs for a split
            continue
        tune, test = d[d.target.isin(TUNE)], d[d.target.isin(TEST)]
        cut = tune.loc[tune.same, col].dropna().quantile(0.9)
        if col == "combined_ratio":   # ranks must come from the tune signs only
            r_ratio = tune.ratio.dropna().sort_values().values
            r_old = tune.old.dropna().sort_values().values
            import numpy as np
            pr = lambda a, x: np.searchsorted(a, x, "right") / len(a)  # noqa: E731
            comb = lambda df: (pr(r_ratio, df.ratio.values) + pr(r_old, df.old.fillna(0).values)) / 2  # noqa: E731
            cut = np.quantile(comb(tune[tune.same]), 0.9)
            fa = float((comb(test[test.same]) > cut).mean())
            ca = float((comb(test[test.differs & ~test.same]) > cut).mean())
        else:
            fa = float((test.loc[test.same, col].dropna() > cut).mean())
            ca = float((test.loc[test.differs & ~test.same, col].dropna() > cut).mean())
        res[col] = {"test_false_alarm": round(fa, 3), "test_catches": round(ca, 3),
                    "reliable": fa <= 0.10 and ca >= 0.50,
                    "n_test_correct": int(test.same.sum()), "n_test_differs": int((test.differs & ~test.same).sum())}
    out[param] = res
(B / "vector_split_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
