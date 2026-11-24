#!/usr/bin/env python3
"""How good is Mouna's judgement? Measure every check against an answer key.

Answer key: ASL-LEX 2.0 phonology (expert coding of each sign's handshape,
location, movement and one/two hands). For each practice sign A:
  - correct attempts: held-out clips of A (never used to build A's
    reference: not human-approved, not rejected) graded against A
  - wrong attempts: held-out clips of every other coded sign B, graded as if
    they were attempts at A. ASL-LEX says which parts truly differ between
    A and B, so each check has a right answer.

Per check we report
  catches      -- of attempts where that part really differs, the share the
                  check flags as wrong
  false alarm  -- of attempts where that part is the same (correct attempts,
                  and wrong signs that share it), the share flagged anyway
  graded       -- the share of attempts the check gave any verdict on
Minimal pairs (signs differing in exactly one part) are reported separately:
they are the cleanest test of whether a check sees *that* part.

Palm orientation is not coded in ASL-LEX, so for it only the false alarm on
correct attempts can be measured.

Writes data/references/judgement_benchmark/summary.json (+ rows.csv.gz).
Run from repo root:  ./venv/bin/python scripts/judgement_benchmark.py [per_correct] [per_wrong]
"""
import csv
import gzip
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app" / "backend")]
OUT = ROOT / "data/references/judgement_benchmark"
PER_CORRECT = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PER_WRONG = int(sys.argv[2]) if len(sys.argv) > 2 else 2
random.seed(7)

# ASL-LEX field -> the part a learner would get wrong
PARTS = {
    "handshape": "Handshape.2.0",
    "location": "MajorLocation.2.0",
    "location_detail": "MinorLocation.2.0",
    "movement": "Movement.2.0",
    "hands": "SignType.2.0",
}
# our checks and the ASL-LEX part each one is meant to catch
CHECKS = {"handshape": "handshape", "location": "location", "hand_count": "hands", "orientation": None}


def asl_lex() -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(open(ROOT / "data/external/asl_lex/signdata.csv", encoding="utf-8", errors="replace")):
        w = re.sub(r"_\d+$", "", r["EntryID"].lower())
        out.setdefault(w, r)
    return out


def code(row: dict, part: str) -> str | None:
    v = (row.get(PARTS[part]) or "").strip()
    if part == "hands":
        return None if not v or v == "NA" else ("one" if v == "OneHanded" else "two")
    return None if not v or v in ("NA", "0") else v


def main() -> None:
    lex = asl_lex()
    from build_sign_references_v2 import load_raw_lookup
    raw = load_raw_lookup()
    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    reviewed = set(dec.video_id.astype(str))          # approved = in a reference; others rejected/unsure
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss", "source"])
    v4["vid"] = v4.video_id.astype(str)
    held = v4[~v4.vid.isin(reviewed) & (v4.source != "openpose_recovered") & v4.vid.map(lambda v: v in raw)]
    pool = {g: list(s.vid) for g, s in held.groupby("gloss")}

    os.chdir(ROOT / "app" / "backend")
    from services.model_service import model_service
    from services.example_videos import practice_ready, all_choices
    model_service.load_model()
    targets = sorted(w for w in all_choices() if practice_ready(w) and w in lex and w in model_service.references)
    others = sorted(g for g in pool if g in lex)
    print(f"{len(targets)} practice signs as targets; {len(others)} coded signs with held-out clips; "
          f"{sum(len(v) for v in pool.values())} held-out clips", flush=True)

    rows = []
    for i, a in enumerate(targets):
        jobs = [(a, v) for v in random.sample(pool.get(a, []), min(PER_CORRECT, len(pool.get(a, []))))]
        for b in others:
            if b != a:
                jobs += [(b, v) for v in random.sample(pool[b], min(PER_WRONG, len(pool[b])))]
        for b, vid in jobs:
            r = model_service.score_against_target(np.asarray(raw[vid], dtype=np.float32).tolist(), a)
            params = r.get("parameters") or {}
            row = {"target": a, "attempt_sign": b, "video_id": vid, "same": a == b,
                   "verdict": r.get("overall_verdict"), "similarity": r.get("embedding_similarity"),
                   "check_hand_count": None if r.get("hand_count_match") is None else (not r["hand_count_match"])}
            for k in ("handshape", "location", "orientation"):
                m = (params.get(k) or {}).get("match")
                row[f"check_{k}"] = None if m is None else (m is False)   # True = flagged as wrong
            for part in PARTS:
                ca, cb = code(lex[a], part), code(lex[b], part)
                row[f"differs_{part}"] = None if ca is None or cb is None else (ca != cb)
            rows.append(row)
        print(f"  {i + 1}/{len(targets)} {a}: {len(jobs)} attempts", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / "rows.csv.gz", "wt") as f:
        df.to_csv(f, index=False)
    summary = summarize(df)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def rate(series) -> dict:
    s = series.dropna().astype(bool)
    return {"rate": round(float(s.mean()), 3) if len(s) else None, "n": int(len(s))}


def summarize(df: pd.DataFrame) -> dict:
    same, diff = df[df.same], df[~df.same]
    out: dict = {"attempts": int(len(df)), "correct_attempts": int(len(same)), "wrong_sign_attempts": int(len(diff))}

    # is it the sign at all? (the overall verdict)
    from sklearn.metrics import roc_auc_score
    sim = df.dropna(subset=["similarity"])
    out["sign"] = {
        "auc": round(float(roc_auc_score(sim.same, sim.similarity)), 3),
        "correct_graded_good": rate(same.verdict.eq("good")),
        "wrong_sign_graded_good": rate(diff.verdict.eq("good")),
    }

    checks = {}
    for check, part in CHECKS.items():
        col = f"check_{check}"
        c = {"graded": round(float(df[col].notna().mean()), 3),
             "false_alarm_on_correct": rate(same[col])}
        if part:
            d = diff[diff[f"differs_{part}"].eq(True)]
            s = diff[diff[f"differs_{part}"].eq(False)]
            c["catches_when_different"] = rate(d[col])
            c["false_alarm_when_same_on_other_signs"] = rate(s[col])
            # minimal pairs: this part differs, the other main parts don't
            main = [p for p in ("handshape", "location", "movement", "hands") if p != part]
            mp = diff[diff[f"differs_{part}"].eq(True) & diff[[f"differs_{p}" for p in main]].eq(False).all(axis=1)]
            c["minimal_pairs_catches"] = rate(mp[col])
        checks[check] = c
    out["checks"] = checks

    # parts we don't check yet: how often signs differ in them (what we're missing)
    out["uncovered_parts"] = {p: rate(diff[f"differs_{p}"]) for p in ("movement", "location_detail")}

    per = []
    for t, g in df.groupby("target"):
        gs, gd = g[g.same], g[~g.same]
        per.append({"sign": t, "correct_good": rate(gs.verdict.eq("good"))["rate"],
                    "wrong_good": rate(gd.verdict.eq("good"))["rate"], "n_correct": int(len(gs))})
    out["per_sign"] = per
    return out


if __name__ == "__main__":
    main()
