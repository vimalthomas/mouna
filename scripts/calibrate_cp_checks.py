#!/usr/bin/env python3
"""Choose how far outside the fluent range counts as "different", per check
family -- without fitting the numbers we report.

Signs are split at random into a TUNE half and a TEST half. On TUNE signs
only, each family's margin is the one that catches the most real differences
(per ASL-LEX) while wrongly flagging at most MAX_FALSE_ALARM of correct
signing. The chosen margins are then scored on the TEST signs, which played
no part in choosing them; those are the numbers to trust.

Families: fingers (+ thumb) vs ASL-LEX handshape, location vs minor location,
movement (strokes/travel) vs movement or repetition.
Writes data/references/cp_tolerances.json (chosen margins + test scores).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.cp_profile import DEFAULT_TOL, compare, features  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402

MAX_FALSE_ALARM = 0.10
GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FAMILIES = {
    "fingers": {"keys": ("finger", "thumb"), "part": lambda fs: any(f["part"] == "handshape" for f in fs), "lex": ("Handshape.2.0",)},
    "location": {"keys": ("location",), "part": lambda fs: any(f["part"] == "location" for f in fs), "lex": ("MinorLocation.2.0",)},
    "movement": {"keys": ("travel",), "part": lambda fs: any(f["feature"] in ("strokes", "travel", "along_palm", "dir_x", "dir_y", "dir_z") for f in fs),
                 "lex": ("Movement.2.0", "RepeatedMovement.2.0")},
}


def main() -> None:
    os.chdir(ROOT)
    lex = {}
    for r in csv.DictReader(open(ROOT / "data/external/asl_lex/signdata.csv", encoding="utf-8", errors="replace")):
        lex.setdefault(re.sub(r"_\d+$", "", r["EntryID"].lower()), r)
    profiles = json.loads((ROOT / "data/references/cp_profiles.json").read_text())
    rows = pd.read_csv(ROOT / "data/references/judgement_benchmark/rows.csv.gz")
    raw = load_raw_lookup()
    feats = {}
    for vid in rows.video_id.astype(str).unique():
        if vid in raw:
            feats[vid] = features(canonicalize_handedness(np.asarray(raw[vid], dtype=np.float32), target="right"))

    def val(r, c):
        x = (r.get(c) or "").strip()
        return None if not x or x == "NA" else x

    items = []
    for r in rows.itertuples():
        f, prof, a, b = feats.get(str(r.video_id)), profiles.get(r.target), lex.get(r.target), lex.get(r.attempt_sign)
        if f is None or not prof or not a or not b:
            continue
        diffs = {}
        for fam, spec in FAMILIES.items():
            pairs = [(val(a, c), val(b, c)) for c in spec["lex"]]
            diffs[fam] = None if any(x is None or y is None for x, y in pairs) else any(x != y for x, y in pairs)
        items.append((r.target, bool(r.same), f, prof, diffs))

    signs = sorted({t for t, *_ in items})
    random.Random(3).shuffle(signs)
    tune = set(signs[: len(signs) // 2])

    def score(fam, m, subset):
        tol = {**DEFAULT_TOL, **{k: m for k in FAMILIES[fam]["keys"]}}
        fa, ca = [], []
        for t, same, f, prof, diffs in items:
            if (t in tune) != (subset == "tune"):
                continue
            flag = FAMILIES[fam]["part"](compare(f, prof, t, tol))
            if same:
                fa.append(flag)
            elif diffs[fam]:
                ca.append(flag)
        return (float(np.mean(fa)) if fa else None, float(np.mean(ca)) if ca else None, len(fa), len(ca))

    chosen, report = {}, {}
    for fam in FAMILIES:
        grid = [(m, *score(fam, m, "tune")) for m in GRID]
        ok = [g for g in grid if g[1] is not None and g[1] <= MAX_FALSE_ALARM]
        m = max(ok, key=lambda g: g[2])[0] if ok else GRID[-1]
        for k in FAMILIES[fam]["keys"]:
            chosen[k] = m
        fa, ca, nf, nc = score(fam, m, "test")
        report[fam] = {"margin": m, "tune_grid": [{"margin": g[0], "false_alarm": round(g[1], 3), "catches": round(g[2], 3)} for g in grid],
                       "test": {"false_alarm_on_correct": round(fa, 3), "catches_when_different": round(ca, 3),
                                "n_correct": nf, "n_different": nc}}
        print(f"{fam:9s} margin {m:<4}  TEST: false alarm {fa:.0%} (n={nf})  catches {ca:.0%} (n={nc})")
    out = {"chosen": chosen, "max_false_alarm": MAX_FALSE_ALARM, "tune_signs": sorted(tune),
           "test_signs": sorted(set(signs) - tune), "families": report}
    (ROOT / "data/references/cp_tolerances.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
