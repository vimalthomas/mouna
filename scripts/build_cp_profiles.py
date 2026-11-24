#!/usr/bin/env python3
"""Build each sign's CP profile: the range fluent signers use for every
measurable part (mouna.coach.cp_profile), from all its approved clips.

Writes data/references/cp_profiles.json  {sign: {feature: {p10,p50,p90,n}}}
Run from repo root:  ./venv/bin/python scripts/build_cp_profiles.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_sign_references_v2 import load_raw_lookup  # noqa: E402
from mouna.coach.cp_profile import features, profile  # noqa: E402
from mouna.data.handedness import canonicalize_handedness  # noqa: E402

OUT = ROOT / "data/references/cp_profiles.json"


def main() -> None:
    import os
    os.chdir(ROOT)
    raw = load_raw_lookup()
    dec = pd.read_csv(ROOT / "data/silver/qa/review_decisions.csv")
    ok = set(dec[dec.decision == "ok"].video_id.astype(str))
    v4 = pd.read_parquet(ROOT / "data/silver/keypoints/keypoints_official_split_v4canon.parquet", columns=["video_id", "gloss"])
    v4["vid"] = v4.video_id.astype(str)
    variants = json.loads((ROOT / "data/references/sign_variants.json").read_text()) if (ROOT / "data/references/sign_variants.json").exists() else {}
    out = {}
    for sign, sub in v4[v4.vid.isin(ok)].groupby("gloss"):
        drop = {i for alt in variants.get(sign, {}).get("variations", []) for i in alt["video_ids"]}  # main form only
        feats = []
        for vid in sub.vid:
            if vid in drop or vid not in raw:
                continue
            f = features(canonicalize_handedness(np.asarray(raw[vid], dtype=np.float32), target="right"))
            if f:
                feats.append(f)
        if len(feats) >= 4:
            out[sign] = profile(feats)
            out[sign]["clips"] = len(feats)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} sign profiles -> {OUT.relative_to(ROOT)}; clips per sign: "
          f"min {min(p['clips'] for p in out.values())}, median {int(np.median([p['clips'] for p in out.values()]))}")
    p = out.get("paper")
    if p:
        print("PAPER:", {k: (round(v['p10'], 2), round(v['p90'], 2)) for k, v in p.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
