"""Which checks are trustworthy enough to tell a learner they're wrong.

Each part check (hand count, handshape, location, palm direction) is scored
by scripts/judgement_benchmark.py against ASL-LEX: how often it wrongly flags
correct signing (false alarm) and how many real differences it catches. A
check is "reliable" only if it passes both bars; otherwise its mismatches are
shown as gentle hints ("worth a look"), never as "fix this", and a hand-count
mismatch can't veto a good verdict. Re-running the benchmark updates this.
"""
import json
from functools import lru_cache
from pathlib import Path

SUMMARY = Path(__file__).resolve().parents[3] / "data/references/judgement_benchmark/summary.json"
MAX_FALSE_ALARM = 0.10   # wrongly flags at most 1 in 10 correct attempts
MIN_CATCHES = 0.50       # catches at least half of real differences (where measurable)
CHECKS = ("hand_count", "handshape", "location", "orientation")


@lru_cache(maxsize=1)
def check_reliability(path: str = str(SUMMARY)) -> dict[str, dict]:
    """{check: {"false_alarm", "catches", "reliable"}}. With no benchmark on
    disk every check counts as reliable (the behaviour before measuring)."""
    try:
        checks = json.loads(Path(path).read_text())["checks"]
    except (OSError, ValueError, KeyError):
        return {c: {"false_alarm": None, "catches": None, "reliable": True} for c in CHECKS}
    out = {}
    for c in CHECKS:
        m = checks.get(c, {})
        fa = (m.get("false_alarm_on_correct") or {}).get("rate")
        ca = (m.get("catches_when_different") or {}).get("rate")
        reliable = fa is not None and fa <= MAX_FALSE_ALARM and (ca is None or ca >= MIN_CATCHES)
        out[c] = {"false_alarm": fa, "catches": ca, "reliable": reliable}
    return out
