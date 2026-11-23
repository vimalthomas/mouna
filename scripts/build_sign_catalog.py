#!/usr/bin/env python3
"""Build data/references/sign_catalog.json: for every sign the app knows
(app vocabulary + EXTRA_WORDS), its ASL-CDI category and ASL-LEX frequency,
and whether it counts as "essential".

Essential = on the ASL-CDI 2.0 inventory (Caselli, Lieberman & Pyers 2020)
AND ASL-LEX 2.0 frequency >= ESSENTIAL_MIN_FREQUENCY (Sehyr et al. 2021).
Categories are the ASL-CDI's own semantic categories, as coded in ASL-LEX;
signs not on the CDI fall under "Other signs". This replaces the app's
hand-made categories so the dictionary is grouped by published work.

Run from the repo root:  ./venv/bin/python scripts/build_sign_catalog.py
"""
import csv
import json
import re
from pathlib import Path

ASL_LEX = Path("data/external/asl_lex/signdata.csv")
SIGNS_JSON = Path("app/backend/signs.json")
OUTPUT = Path("data/references/sign_catalog.json")
ESSENTIAL_MIN_FREQUENCY = 5.0
# Essential signs outside the 100-sign model vocabulary (ASL Citizen clips).
EXTRA_WORDS = ["sit", "ask", "dry", "small", "not", "why", "where", "when"]


def main() -> None:
    by_word: dict[str, list[dict]] = {}
    for row in csv.DictReader(open(ASL_LEX, encoding="utf-8", errors="replace")):
        key = re.sub(r"_\d+$", "", row["EntryID"].lower())  # apple, apple_2 -> apple
        by_word.setdefault(key, []).append(row)

    vocab = [s["word"] for s in json.load(open(SIGNS_JSON))["signs"]]
    catalog = {}
    for word in vocab + [w for w in EXTRA_WORDS if w not in vocab]:
        rows = by_word.get(word.replace(" ", "_")) or by_word.get(word)
        if not rows:
            catalog[word] = {"in_asl_lex": False, "in_cdi": False, "category": "Other signs",
                             "frequency": None, "essential": False, "in_model_vocab": word in vocab}
            continue
        row = max(rows, key=lambda r: float(r["SignFrequency(M)"] or 0))  # most frequent entry
        freq = float(row["SignFrequency(M)"]) if row["SignFrequency(M)"] else None
        in_cdi = row.get("InCDI", "").strip() == "Yes"
        catalog[word] = {
            "in_asl_lex": True,
            "in_cdi": in_cdi,
            "category": (row.get("CDISemanticCategory") or "").strip() if in_cdi else "Other signs",
            "frequency": round(freq, 2) if freq is not None else None,
            "essential": bool(in_cdi and freq is not None and freq >= ESSENTIAL_MIN_FREQUENCY),
            "in_model_vocab": word in vocab,
        }

    OUTPUT.write_text(json.dumps(catalog, indent=2, sort_keys=True))
    ess = [w for w, c in catalog.items() if c["essential"]]
    print(f"{len(catalog)} signs, {len(ess)} essential -> {OUTPUT}")


if __name__ == "__main__":
    main()
