#!/usr/bin/env python3
"""Turn the variant proposals (discover_variants.py) into the hand-count
rule agreed with the project owner:

  The way MOST approved clips sign it (one hand or two) is the sign's main
  form -- used for the reference, model training, the example, the 3D hands
  and grading. The other form is shown on the Learn screen as a variation.

Only signs whose clusters differ in hand count are affected; per-clip hand
counts alone are not used, because the tracker misses pressed-together
hands (bed, cook, school) and would wrongly vote those one-handed.

Writes data/references/sign_variants.json:
  {sign: {"primary_hands": n, "primary_video_ids": [...],
          "variations": [{"n_hands": m, "video_ids": [...], "share": f}]}}
Run: ./venv/bin/python scripts/assign_variants.py
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSALS = ROOT / "data/references/variant_proposals.json"
OUT = ROOT / "data/references/sign_variants.json"


def assign(proposals: dict) -> dict:
    out = {}
    for word, p in proposals.items():
        if "hands" not in p.get("differs_in", []):
            continue
        by_hands = defaultdict(list)
        for c in p["clusters"]:
            by_hands[c["n_hands"]].extend(c["video_ids"])
        total = sum(len(v) for v in by_hands.values())
        # majority wins; a tie keeps two hands (the tracker under-counts hands)
        primary = max(by_hands, key=lambda n: (len(by_hands[n]), n))
        out[word] = {
            "primary_hands": primary,
            "primary_video_ids": sorted(by_hands[primary]),
            "variations": [{"n_hands": n, "video_ids": sorted(v), "share": round(len(v) / total, 2)}
                           for n, v in sorted(by_hands.items()) if n != primary],
        }
    return out


def main() -> None:
    out = assign(json.loads(PROPOSALS.read_text()))
    OUT.write_text(json.dumps(out, indent=2))
    for w, v in out.items():
        alt = ", ".join(f"{x['n_hands']}-handed {x['share']:.0%}" for x in v["variations"])
        print(f"  {w:10s} main: {v['primary_hands']}-handed ({len(v['primary_video_ids'])} clips); variation: {alt}")
    print(f"Saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
