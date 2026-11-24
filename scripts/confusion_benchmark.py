#!/usr/bin/env python3
"""Score explain_confusion(): when an attempt of sign B is graded as sign A,
does it name a part where A and B really differ (per ASL-LEX)? And for
correct attempts at A, how often does it wrongly claim "this looked like B"?
Writes data/references/judgement_benchmark/confusion_summary.json
"""
import csv, json, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.cp_profile import explain_confusion, features  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402

PART_COLS = {"movement": ("Movement.2.0", "RepeatedMovement.2.0"), "location": ("MinorLocation.2.0", "MajorLocation.2.0"),
             "handshape": ("Handshape.2.0", "SelectedFingers.2.0", "Flexion.2.0")}


def main():
    os.chdir(ROOT)
    lex = {}
    for r in csv.DictReader(open(ROOT / "data/external/asl_lex/signdata.csv", encoding="utf-8", errors="replace")):
        lex.setdefault(re.sub(r"_\d+$", "", r["EntryID"].lower()), r)
    prof = json.loads((ROOT / "data/references/cp_profiles.json").read_text())
    rows = pd.read_csv(ROOT / "data/references/judgement_benchmark/rows.csv.gz")
    raw = load_raw_lookup()
    cache = {}
    def feats(v):
        if v not in cache:
            cache[v] = features(canonicalize_handedness(np.asarray(raw[v], dtype=np.float32), target="right")) if v in raw else None
        return cache[v]
    def val(r, c):
        x = (r.get(c) or "").strip(); return None if not x or x == "NA" else x
    named, right, fa_correct, n_correct = 0, 0, 0, 0
    wrong = rows[~rows.same.astype(bool) & rows.target.isin(prof) & rows.attempt_sign.isin(prof)]
    # the confusable cases: other signs that came closest (top 30% similarity per target)
    wrong = wrong[wrong.similarity >= wrong.groupby("target").similarity.transform(lambda s: s.quantile(0.7))]
    for r in wrong.itertuples():
        f = feats(str(r.video_id))
        if f is None or r.target not in lex or r.attempt_sign not in lex:
            continue
        ex = explain_confusion(f, prof[r.target], prof[r.attempt_sign], r.target, r.attempt_sign)
        if ex:
            named += 1
            cols = PART_COLS[ex[0]["part"]]
            right += any(val(lex[r.target], c) != val(lex[r.attempt_sign], c) for c in cols if val(lex[r.target], c) and val(lex[r.attempt_sign], c))
    others = wrong.groupby("target").attempt_sign.apply(lambda s: list(dict.fromkeys(s))[:5]).to_dict()
    for r in rows[rows.same.astype(bool) & rows.target.isin(prof)].itertuples():
        f = feats(str(r.video_id))
        for o in others.get(r.target, []):
            if f is None:
                continue
            n_correct += 1
            fa_correct += bool(explain_confusion(f, prof[r.target], prof[o], r.target, o))
    out = {"wrong_attempts": int(len(wrong)), "named_a_part": round(named / max(len(wrong), 1), 3),
           "named_part_really_differs": round(right / max(named, 1), 3),
           "false_claim_on_correct": round(fa_correct / max(n_correct, 1), 3), "n_correct_checks": n_correct}
    (ROOT / "data/references/judgement_benchmark/confusion_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
