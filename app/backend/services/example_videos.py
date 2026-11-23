"""Per-sign example-video choices from scripts/pick_example_videos.py: the
video whose signing agrees with the reference used for grading, plus that
video's own hand bbox / hands-up window. Re-read when the file changes, so
re-running the script takes effect without a restart."""

import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[3] / "data" / "references" / "example_videos.json"

_cache: dict = {}
_mtime: float | None = None


def all_choices() -> dict:
    global _cache, _mtime
    try:
        m = PATH.stat().st_mtime
    except FileNotFoundError:
        return {}
    if m != _mtime:
        _cache = json.loads(PATH.read_text())
        _mtime = m
    return _cache


def get(word: str) -> dict | None:
    return all_choices().get(word.lower())


# A sign is offered for practice only when its shown example agrees with
# the grading reference clearly: good verdict, same number of hands, all
# three CP parts matching, and strong embedding similarity. Borderline
# signs (e.g. "can": 1/3 parts; "decide": 0.67) would make the coach's
# per-part feedback contradict the very video the learner copies.
MIN_SIMILARITY = 0.8


def practice_ready(word: str) -> bool:
    c = get(word)
    return bool(
        c and c.get("verdict") == "good" and c.get("hand_count_match")
        and c.get("params_matched") == 3 and (c.get("similarity") or 0) >= MIN_SIMILARITY
    )
