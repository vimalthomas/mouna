"""Live five-parameter scores for an attempt (the Score tab).

Handshape, location and movement: the attempt's vector (joint angles,
position relative to the face/body, time-aligned movement path) is compared
with the fluent signers of the sign (mouna.coach.vectors), and combined with
the CP deviation (mouna.coach.scores) by rank. Levels and the reliable flag
come from scripts/build_vector_references.py (cuts chosen on tune signs,
measured on held-out signs). Palm orientation: the CP score only (ASL-LEX
can't check a vector version). Facial expression: not measured yet.
"""
import math
import pickle
from pathlib import Path

import numpy as np

from mouna.coach import cp_profile
from mouna.coach.scores import LABELS, PARAMETERS, calibration, score_parameters
from mouna.coach.vectors import HS_NAMES, LOC_NAMES, handshape_vector, location_vector, movement_path

REFS = Path(__file__).resolve().parents[3] / "data/references/vector_references.pkl"
_REFS: dict | None = None
_PROFILES: dict | None = None
VECTOR_FN = {"handshape": handshape_vector, "location": location_vector}
UNITS = {"handshape": "°", "location": " shoulder widths"}
NAMES = {"handshape": HS_NAMES, "location": LOC_NAMES}


def _refs() -> dict:
    global _REFS
    if _REFS is None:
        try:
            with open(REFS, "rb") as f:
                _REFS = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError):
            _REFS = {"clouds": {}, "calibration": {}}
    return _REFS


def _profiles() -> dict:
    global _PROFILES
    if _PROFILES is None:
        import json
        p = Path(__file__).resolve().parents[3] / "data/references/cp_profiles.json"
        _PROFILES = json.loads(p.read_text()) if p.exists() else {}
    return _PROFILES


def _pct(arr: np.ndarray, x: float) -> float:
    return float((np.searchsorted(arr, x, "left") + np.searchsorted(arr, x, "right")) / 2 / len(arr))


def _level(comb: float, cal: dict) -> tuple[str, int]:
    fc, nc = cal["fluent_cut"], cal["needs_work_cut"]
    level = "fluent" if comb <= fc else "close" if comb <= nc else "needs_work"
    return level, int(round(100 * math.exp(-1.4 * max(0.0, comb - fc) / max(nc - fc, 0.05))))


def parameter_scores(canon: np.ndarray, target: str, width: int = 640, height: int = 480) -> dict | None:
    """{param: {label, level, score, reliable, method, evidence, ...}} or None
    when the sign has no fluent profile. `canon`: canonicalised raw keypoints (T, 225)."""
    prof = _profiles().get(target)
    feats = cp_profile.features(canon)
    if prof is None or feats is None:
        return None
    cp = score_parameters(feats, prof, calibration())
    refs = _refs()
    out = {}
    for p in PARAMETERS:
        base = cp[p]
        entry = {"label": LABELS[p], "level": base["level"], "score": base["score"], "reliable": bool(base["reliable"]),
                 "method": "cp" if base["level"] else "none", "evidence": [], "cp_worst": base["worst"]}
        cloud = refs["clouds"].get(p, {}).get(target)
        cal = refs["calibration"].get(p)
        if cloud is not None and cal is not None:
            if p == "movement":
                path = movement_path(canon, width, height)
                ratio = float(cloud.ratio(path[None])[0]) if path is not None else None
                vec = None
            else:
                vec = VECTOR_FN[p](canon, width, height)
                ratio = cloud.distance(vec) / max(float(np.median(cloud.null)), 1e-9) if vec is not None else None
            if ratio is not None:
                dev = base["deviation"] or 0.0
                comb = (_pct(cal["ratio_ranks"], ratio) + _pct(cal["cp_ranks"], dev)) / 2
                level, score = _level(comb, cal)
                entry.update(level=level, score=score, reliable=bool(cal["reliable"]), method="vector+cp",
                             ratio=round(ratio, 2), combined=round(comb, 3), n_fluent=int(len(cloud.null)))
                if vec is not None and level != "fluent":
                    entry["evidence"] = [{**e, "unit": UNITS[p]} for e in cloud.evidence(vec, NAMES[p], top=3)]
        out[p] = entry
    return out
