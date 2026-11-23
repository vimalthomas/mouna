#!/usr/bin/env python3
"""Score the CP-profile checks (mouna.coach.cp_profile) on the judgement
benchmark's held-out attempts -- the same yardsticks as judgement_benchmark.py:
catches when ASL-LEX says the part differs, false alarms on correct signing.

Profiles come from approved clips; every benchmark attempt is a clip that was
never approved, so nothing here was seen when building the profiles.

Checks scored:
  fingers   any finger or thumb outside the fluent range  vs  ASL-LEX handshape
  location  hand position vs the nose outside the range    vs  ASL-LEX minor location
  movement  strokes / travel outside the range             vs  ASL-LEX movement or repetition
  contact   hands didn't touch where fluent signers' do    vs  ASL-LEX contact
Writes data/references/judgement_benchmark/profile_summary.json
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.cp_profile import compare, features  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402

OUT = ROOT / "data/references/judgement_benchmark"


def lex() -> dict:
    out = {}
    for r in csv.DictReader(open(ROOT / "data/external/asl_lex/signdata.csv", encoding="utf-8", errors="replace")):
        out.setdefault(re.sub(r"_\d+$", "", r["EntryID"].lower()), r)
    return out


def main() -> None:
    import os
    os.chdir(ROOT)
    L = lex()
    profiles = json.loads((ROOT / "data/references/cp_profiles.json").read_text())
    rows = pd.read_csv(OUT / "rows.csv.gz")
    raw = load_raw_lookup()
    cache = {}
    def feats(vid):
        if vid not in cache:
            cache[vid] = features(canonicalize_handedness(np.asarray(raw[vid], dtype=np.float32), target="right")) if vid in raw else None
        return cache[vid]

    def v(r, c):
        x = (r.get(c) or "").strip()
        return None if not x or x == "NA" else x

    out = []
    for r in rows.itertuples():
        prof = profiles.get(r.target)
        f = feats(str(r.video_id))
        a, b = L.get(r.target), L.get(r.attempt_sign)
        if not prof or f is None or not a or not b:
            continue
        parts = {x["part"] for x in compare(f, prof, r.target)}
        contact_flag = any(x["feature"] == "gap" for x in compare(f, prof, r.target))
        def differs(*cols):
            vals = [(v(a, c), v(b, c)) for c in cols]
            if any(x is None or y is None for x, y in vals):
                return None
            return any(x != y for x, y in vals)
        out.append({"same": r.same, "target": r.target,
                    "fingers": "handshape" in parts, "location": "location" in parts,
                    "movement": "movement" in parts and not contact_flag or any(
                        x["feature"] in ("strokes", "travel") for x in compare(f, prof, r.target)),
                    "contact": contact_flag,
                    "d_fingers": differs("Handshape.2.0"),
                    "d_location": differs("MinorLocation.2.0"),
                    "d_movement": differs("Movement.2.0", "RepeatedMovement.2.0"),
                    "d_contact": differs("Contact.2.0"),
                    "d_other": differs("Handshape.2.0", "MinorLocation.2.0", "Movement.2.0", "RepeatedMovement.2.0", "Contact.2.0")})
    d = pd.DataFrame(out)
    same, wrong = d.same.astype(bool), ~d.same.astype(bool)

    def rate(s):
        s = s.dropna().astype(bool)
        return {"rate": round(float(s.mean()), 3) if len(s) else None, "n": int(len(s))}

    summary = {"attempts": int(len(d)), "correct": int(same.sum())}
    for c in ("fingers", "location", "movement", "contact"):
        dc = d[f"d_{c}"]
        summary[c] = {"false_alarm_on_correct": rate(d.loc[same, c]),
                      "catches_when_different": rate(d.loc[wrong & dc.eq(True), c]),
                      "false_alarm_when_same_on_other_signs": rate(d.loc[wrong & dc.eq(False), c])}
    (OUT / "profile_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
