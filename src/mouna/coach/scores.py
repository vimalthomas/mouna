"""Scores on the five parameters of a sign, built from the measured CPs.

Every CP feature in mouna.coach.cp_profile belongs to one of the five
phonological parameters of ASL:
  handshape    finger_* (each finger's straightness), thumb_out
  location     loc_x, loc_y (hand position relative to the nose)
  orientation  palm_x, palm_y, palm_z (palm direction)
  movement     dir_x, dir_y (main motion axis), strokes, travel, gap (contact)
  nonmanual    facial expression -- not measured yet
For each CP: how far the learner's value sits outside the range fluent
signers use (10th-90th percentile of the sign's approved clips), in units of
that range's typical spread; inside the range = 0. A parameter's deviation is
its worst CP (one bent finger is enough to make a handshape wrong).

Levels: up to where 80% of correct signers fall -> "fluent"; up to the
cut-off that flags at most 10% of correct signing -> "close"; beyond ->
"needs_work". The cut-offs, and how well each parameter separates real
differences from correct signing, come from scripts/calibrate_scores.py
(chosen on half the signs, measured on the other half) and are stored in
data/references/score_calibration.json.
"""
import json
import math
from pathlib import Path

PARAMETERS = {
    "handshape": ["finger_thumb", "finger_index", "finger_middle", "finger_ring", "finger_little", "thumb_out"],
    "location": ["loc_x", "loc_y"],
    "orientation": ["palm_x", "palm_y", "palm_z"],
    "movement": ["dir_x", "dir_y", "strokes", "travel", "gap"],
    "nonmanual": [],
}
LABELS = {"handshape": "Handshape", "location": "Location", "orientation": "Palm orientation",
          "movement": "Movement", "nonmanual": "Facial expression"}
CALIBRATION = Path(__file__).resolve().parents[3] / "data/references/score_calibration.json"
DEFAULT_CUTOFF = 1.0


def _spread(r: dict) -> float:
    return max((r["p90"] - r["p10"]) / 2, 0.05)


def cp_deviation(feature: str, x, r: dict | None, prof: dict) -> float | None:
    """How unusual the learner's value is for a fluent signer (0 = inside the range)."""
    if r is None:
        return None
    if feature == "gap":
        contact_sign = prof.get("two_handed_share", 0) >= 0.6 and r["p90"] < 1.2
        if not contact_sign:
            return None                      # contact isn't part of this sign
        if x is None:
            return 3.0                       # one hand only, in a sign where the hands touch
    if x is None:
        return None
    s = _spread(r) if feature != "strokes" else max((r["p90"] - r["p10"]) / 2, 1.0)
    if x < r["p10"]:
        return (r["p10"] - x) / s
    if x > r["p90"]:
        return (x - r["p90"]) / s
    return 0.0


def calibration() -> dict:
    try:
        return json.loads(CALIBRATION.read_text())
    except (OSError, ValueError):
        return {}


def score_parameters(att: dict, prof: dict, cal: dict | None = None) -> dict:
    """{parameter: {"level", "score" 0-100, "deviation", "worst", "cps": [...],
    "reliable", "auc"}} -- "level" None when the parameter isn't measured."""
    cal = cal if cal is not None else calibration()
    out = {}
    for p, feats in PARAMETERS.items():
        cps = []
        for f in feats:
            r = prof.get(f)
            d = cp_deviation(f, att.get(f), r, prof)
            if d is None:
                continue
            cps.append({"cp": f, "yours": att.get(f), "p10": r["p10"], "p50": r["p50"], "p90": r["p90"], "deviation": round(d, 3)})
        info = cal.get("parameters", {}).get(p, {})
        if not cps:
            out[p] = {"level": None, "score": None, "deviation": None, "worst": None, "cps": [],
                      "reliable": False, "auc": info.get("test_auc")}
            continue
        worst = max(cps, key=lambda c: c["deviation"])
        d = worst["deviation"]
        cut = info.get("cutoff", DEFAULT_CUTOFF)
        fluent = info.get("fluent_cutoff", 0.0)
        level = "fluent" if d <= fluent else "close" if d <= cut else "needs_work"
        # 100 up to where most fluent signers are; halves for each further step of the cut-off's scale
        excess = max(0.0, d - fluent) / max(cut - fluent, 0.5)
        out[p] = {"level": level, "score": int(round(100 * math.exp(-1.4 * excess))), "deviation": round(d, 3),
                  "worst": worst["cp"] if d > 0 else None, "cps": cps,
                  "reliable": bool(info.get("reliable", False)), "auc": info.get("test_auc")}
    return out
