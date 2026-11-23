#!/usr/bin/env python3
"""Focused review queue: only what's still needed to bring the ESSENTIAL
signs (data/references/sign_catalog.json) into the practice set.

The full queue (review_queue.json) walks all 100 signs alphabetically and
every clip of each (60+ for some), but a sign only needs TARGET_OK approved
clips to build a reference. This queue:
  - covers essential signs the model is trained on (in the review queue),
  - skips clips that already have a decision,
  - per sign, asks for only the clips still needed + SPARE (some get rejected),
  - picks each sign's most promising clips first: no automatic QA flags,
    the cross-validated model agrees with the label, a playable video, good
    hand tracking,
  - orders signs by ASL-LEX frequency (most used first).
Re-run any time: signs drop out as they reach TARGET_OK approvals.

Run from repo root:
    ./venv/bin/python scripts/qa_build_essential_queue.py
Then review with it:
    MOUNA_QA_QUEUE=data/silver/qa/review_queue_focus_essential.json PYTHONPATH=src ./venv/bin/python scripts/qa_review_server.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

FULL_QUEUE = Path("data/silver/qa/review_queue.json")
DECISIONS = Path("data/silver/qa/review_decisions.csv")
CATALOG = Path("data/references/sign_catalog.json")
OUT = Path("data/silver/qa/review_queue_focus_essential.json")
TARGET_OK = 8
SPARE = 3


def promise(item: dict) -> tuple:
    conf = item.get("confidence") or {}
    return (
        not item.get("flags"),                      # no automatic QA flags
        bool(conf.get("agrees")),                   # model agrees with the label
        item.get("video_path") is not None,         # can actually be watched
        conf.get("claimed_prob") or 0.0,            # model's belief in the label
        item.get("hand_detect_rate") or 0.0,        # hands visible
    )


def main() -> None:
    catalog = json.loads(CATALOG.read_text())
    essential = {w: c for w, c in catalog.items() if c.get("essential")}
    queue = json.loads(FULL_QUEUE.read_text())
    decided = {}
    if DECISIONS.exists():
        decided = {r["video_id"]: r["decision"] for r in csv.DictReader(open(DECISIONS))}

    gloss_of = {str(i["video_id"]): i["gloss"] for i in queue}
    ok_count = Counter(gloss_of[v] for v, d in decided.items() if d == "ok" and v in gloss_of)

    by_sign: dict[str, list[dict]] = {}
    for item in queue:
        if item["gloss"] in essential and str(item["video_id"]) not in decided:
            by_sign.setdefault(item["gloss"], []).append(item)

    out, summary = [], []
    for sign in sorted(by_sign, key=lambda w: -(essential[w].get("frequency") or 0)):
        need = TARGET_OK - ok_count[sign]
        if need <= 0:
            continue
        picks = sorted(by_sign[sign], key=promise, reverse=True)[: need + SPARE]
        out += [{**p, "reason": f"essential: {ok_count[sign]}/{TARGET_OK} approved"} for p in picks]
        summary.append((sign, ok_count[sign], len(picks)))

    OUT.write_text(json.dumps(out, indent=2))
    done = sorted(w for w in essential if ok_count[w] >= TARGET_OK)
    print(f"{len(out)} clips across {len(summary)} signs -> {OUT}")
    print(f"Already at {TARGET_OK}+ approvals: {', '.join(done) or 'none'}")
    for sign, have, n in summary:
        print(f"  {sign:10s} approved {have}/{TARGET_OK}, queued {n}")


if __name__ == "__main__":
    main()
