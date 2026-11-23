"""Measure a signer's CPs directly, and compare them with the range fluent
signers use -- not with one reference clip.

features(seq) measures one clip (canonical keypoints, dominant hand in the
right slot):
  finger_<name>   how straight each finger is: fingertip-to-wrist distance /
                  knuckle-to-wrist distance (about 2 straight, about 1 curled)
  thumb_out       thumb tip distance from the index knuckle / hand size
  palm_x/y/z      which way the palm faces (unit normal, image axes)
  loc_x/loc_y     where the hand is, relative to the nose, in shoulder widths
  travel          how far the dominant wrist moved (shoulder widths)
  strokes         back-and-forth strokes (1 = once)
  gap             closest the two hands came (hand sizes; None if one hand)
A sign's profile is the 10th-90th percentile of each feature over its
approved clips (scripts/build_cp_profiles.py). compare() flags a feature only
when an attempt falls clearly outside that range, and says what to do in
plain words. Every finding is a hint until the judgement benchmark shows the
underlying check is reliable (mouna.coach.reliability).
"""
import json
from pathlib import Path

import numpy as np

FINGERS = {"thumb": (1, 4), "index": (5, 8), "middle": (9, 12), "ring": (13, 16), "little": (17, 20)}
R, L = slice(54, 75), slice(33, 54)


def _hand_frames(a: np.ndarray):
    r = a[:, R]
    present = np.any(r[:, :, :2] != 0, axis=(1, 2))
    return r, present


def _strokes(w: np.ndarray, min_amp=0.08) -> int:
    if len(w) < 4:
        return 1
    c = w - w.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    x = c @ axis
    turns, last, direction = 0, x[0], 0
    for v in x[1:]:
        d = v - last
        if abs(d) < min_amp:
            continue
        s = 1 if d > 0 else -1
        if direction and s != direction:
            turns += 1
        direction, last = s, v
    return turns // 2 + 1


def features(seq) -> dict | None:
    a = np.asarray(seq, dtype=np.float32).reshape(len(seq), 75, 3)
    r, present = _hand_frames(a)
    sh = a[:, [11, 12], :2]
    ok_sh = np.all(sh != 0, axis=(1, 2))
    if present.sum() < 3 or ok_sh.sum() < 3:
        return None
    # movement is measured in the signing phase only: the middle of the time a
    # hand is up, without the raise from rest and the drop back down (those
    # dominated the "main direction" and hid the sign's own motion)
    up = np.where(present)[0]
    lo, hi = up[0], up[-1]
    trim = int(round((hi - lo) * 0.2))
    core = present.copy()
    core[: lo + trim] = False
    core[hi - trim + 1:] = False
    if core.sum() < 3:
        core = present
    sw = float(np.median(np.linalg.norm(sh[ok_sh, 0] - sh[ok_sh, 1], axis=1))) or 1.0
    h = r[present]                                          # (n, 21, 3)
    size = np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=1)
    size = np.where(size > 1e-6, size, np.nan)
    f: dict = {}
    for name, (base, tip) in FINGERS.items():
        knuckle = np.linalg.norm(h[:, base, :2] - h[:, 0, :2], axis=1)
        reach = np.linalg.norm(h[:, tip, :2] - h[:, 0, :2], axis=1)
        f[f"finger_{name}"] = float(np.nanmedian(reach / np.where(knuckle > 1e-6, knuckle, np.nan)))
    f["thumb_out"] = float(np.nanmedian(np.linalg.norm(h[:, 4, :2] - h[:, 5, :2], axis=1) / size))
    n = np.cross(h[:, 5] - h[:, 0], h[:, 17] - h[:, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    f["palm_x"], f["palm_y"], f["palm_z"] = (float(v) for v in np.median(n, axis=0))
    nose = a[present, 0, :2]
    centre = h[:, [0, 5, 9, 13, 17], :2].mean(1)
    rel = (centre - nose) / sw
    f["loc_x"], f["loc_y"] = (float(v) for v in np.median(rel, axis=0))
    wrist = (r[core][:, 0, :2] - a[core, 0, :2].mean(0)) / sw
    f["travel"] = float(np.linalg.norm(np.diff(wrist, axis=0), axis=1).sum())
    f["strokes"] = float(_strokes(wrist))
    l = a[:, L]
    lp = np.any(l[:, :, :2] != 0, axis=(1, 2))
    both = present & lp
    if both.sum() >= 2:
        gap = np.linalg.norm(a[both][:, R, :2].mean(1) - a[both][:, L, :2].mean(1), axis=1)
        f["gap"] = float(np.nanmin(gap / np.nanmedian(size)))
    else:
        f["gap"] = None
    # main axis of the dominant hand's motion: |x| sideways, |y| up-down, |z| toward the viewer
    w3 = r[core][:, 0, :2]
    if len(w3) >= 4:
        c = w3 - w3.mean(0)
        ax = np.linalg.svd(c, full_matrices=False)[2][0]
        f["dir_x"], f["dir_y"] = (float(abs(v)) for v in ax)
    # two hands: does the moving hand slide ALONG the other palm (PAPER) or move ONTO it (SCHOOL)?
    # share of the motion that lies in the other hand's palm plane (1 = slides, 0 = straight onto it)
    both_core = both & core
    if both_core.sum() >= 4:
        both = both_core
        lb = a[both][:, L]
        nb = np.cross(lb[:, 5] - lb[:, 0], lb[:, 17] - lb[:, 0])
        nb = nb / np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-9)
        n_med = np.median(nb, axis=0)
        n_med = n_med / max(np.linalg.norm(n_med), 1e-9)
        steps = np.diff(a[both][:, R, :3].mean(1), axis=0)
        normal = np.abs(steps @ n_med)
        along = np.linalg.norm(steps - np.outer(steps @ n_med, n_med), axis=1)
        tot = (normal + along).sum()
        f["along_palm"] = float(along.sum() / tot) if tot > 1e-9 else None
    else:
        f["along_palm"] = None
    return f


def profile(clip_features: list[dict]) -> dict:
    """10th-90th percentile range per feature over fluent signers' clips."""
    out = {}
    keys = {k for c in clip_features for k in c}
    for k in keys:
        v = np.array([c[k] for c in clip_features if c.get(k) is not None], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) >= 4:
            out[k] = {"p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
                      "p90": float(np.percentile(v, 90)), "n": int(len(v))}
    out["two_handed_share"] = float(np.mean([c.get("gap") is not None for c in clip_features])) if clip_features else 0.0
    return out


def _outside(x, r, rel=0.25, abs_=0.0):
    """-1 below / +1 above the fluent range (with a margin), else 0."""
    if x is None or r is None:
        return 0
    margin = rel * (r["p90"] - r["p10"]) + abs_
    return -1 if x < r["p10"] - margin else 1 if x > r["p90"] + margin else 0


FINGER_WORDS = {"thumb": "thumb", "index": "index finger", "middle": "middle finger", "ring": "ring finger", "little": "little finger"}


# How far outside the fluent range counts as different, per family: a multiple
# of the range's width (plus a small absolute floor). Chosen by
# scripts/calibrate_cp_checks.py on half the signs, reported on the other half;
# stored in data/references/cp_tolerances.json.
TOL_FILE = Path(__file__).resolve().parents[3] / "data/references/cp_tolerances.json"
DEFAULT_TOL = {"finger": 0.25, "thumb": 0.25, "location": 0.25, "travel": 0.4}


def tolerances() -> dict:
    try:
        return {**DEFAULT_TOL, **json.loads(TOL_FILE.read_text())["chosen"]}
    except (OSError, ValueError, KeyError):
        return dict(DEFAULT_TOL)


def compare(att: dict, prof: dict, sign: str, tol: dict | None = None) -> list[dict]:
    """Findings: {"part", "feature", "say"} -- only where the attempt is clearly
    outside what fluent signers do."""
    t = tol or tolerances()
    S, out = sign.upper(), []
    for name, word in FINGER_WORDS.items():
        k = f"finger_{name}"
        o = _outside(att.get(k), prof.get(k), rel=t["finger"], abs_=0.12)
        if o < 0:
            out.append({"part": "handshape", "feature": k, "say": f"Your {word} is more bent than fluent signers' in {S}; straighten it."})
        elif o > 0:
            out.append({"part": "handshape", "feature": k, "say": f"Your {word} is straighter than fluent signers' in {S}; bend it more."})
    o = _outside(att.get("thumb_out"), prof.get("thumb_out"), rel=t["thumb"], abs_=0.15)
    if o:
        out.append({"part": "handshape", "feature": "thumb_out",
                    "say": f"Your thumb is {'tucked in' if o < 0 else 'sticking out'} more than in {S}."})
    ox = _outside(att.get("loc_x"), prof.get("loc_x"), rel=t["location"], abs_=0.2)
    oy = _outside(att.get("loc_y"), prof.get("loc_y"), rel=t["location"], abs_=0.2)
    if oy:
        out.append({"part": "location", "feature": "loc_y", "say": f"Your hand is {'higher' if oy < 0 else 'lower'} than where {S} is signed."})
    if ox:
        out.append({"part": "location", "feature": "loc_x", "say": f"Your hand is too far {'toward the middle' if ox > 0 else 'to the side'} for {S}."})
    if prof.get("two_handed_share", 0) >= 0.6 and prof.get("gap") and prof["gap"]["p90"] < 1.2:
        if att.get("gap") is None:
            out.append({"part": "movement", "feature": "gap", "say": f"Only one hand was seen; in {S} one hand touches the other."})
        elif _outside(att["gap"], prof["gap"], abs_=0.4) > 0:
            out.append({"part": "movement", "feature": "gap", "say": f"Your hands didn't come together; in {S} one hand touches the other."})
    st = prof.get("strokes")
    if st and att.get("strokes") is not None:
        if st["p10"] >= 2 and att["strokes"] < 2:
            out.append({"part": "movement", "feature": "strokes", "say": f"In {S} the movement repeats; you did it once."})
        elif st["p90"] <= 1 and att["strokes"] >= 3:
            out.append({"part": "movement", "feature": "strokes", "say": f"In {S} the movement happens once; you repeated it."})
    if prof.get("two_handed_share", 0) >= 0.6 and att.get("along_palm") is not None:
        o = _outside(att["along_palm"], prof.get("along_palm"), rel=t["travel"], abs_=0.08)
        if o < 0:
            out.append({"part": "movement", "feature": "along_palm",
                        "say": f"Your hand moved onto the other palm; in {S} it slides along the palm."})
        elif o > 0:
            out.append({"part": "movement", "feature": "along_palm",
                        "say": f"Your hand slid along the other palm; in {S} it moves onto the palm."})
    dirs = {"dir_y": "up and down", "dir_x": "side to side", "dir_z": "toward and away from you"}
    for k, words in dirs.items():
        r_ = prof.get(k)
        if r_ and att.get(k) is not None and r_["p10"] >= 0.6 and att[k] < r_["p10"] - t["travel"] * (r_["p90"] - r_["p10"]) - 0.15:
            out.append({"part": "movement", "feature": k, "say": f"In {S} the hand mainly moves {words}; yours moved in a different direction."})
    o = _outside(att.get("travel"), prof.get("travel"), rel=t["travel"], abs_=0.1)
    if o:
        out.append({"part": "movement", "feature": "travel",
                    "say": f"Your hand moved {'much less' if o < 0 else 'much more'} than fluent signers' in {S}; make the movement {'bigger' if o < 0 else 'smaller'}."})
    return out


# ---- why did it look like another sign? -------------------------------------

FEATURE_WORDS = {
    "dir_x": ("moved mostly side to side", "slides side to side"),
    "dir_y": ("moved mostly up and down", "moves up and down"),
    "strokes": ("repeated the movement", "repeats the movement"),
    "travel": ("made a movement of that size", "has a movement of that size"),
    "loc_y": ("was at that height", "is signed at that height"),
    "loc_x": ("was that far to the side", "is signed that far to the side"),
    "thumb_out": ("held the thumb that way", "holds the thumb that way"),
}
FEATURE_WORDS.update({f"finger_{n}": (f"held the {w} that way", f"holds the {w} that way") for n, w in FINGER_WORDS.items()})


def _spread(r: dict) -> float:
    return max((r["p90"] - r["p10"]) / 2, 0.05)


# min_gap 2.0: chosen on half the signs; on the other half it names a cause in ~72% of
# look-alike attempts and the named part truly differs (ASL-LEX) ~93% of the time.
def explain_confusion(att: dict, prof_target: dict, prof_other: dict, sign: str, other: str,
                      min_gap: float = 2.0, max_points: int = 1) -> list[dict]:
    """The learner's measurements that sit closer to the OTHER sign's typical
    value than to the target's -- i.e. the CP that made it look like `other`.
    Only features where the two signs' fluent signers clearly differ count
    (medians at least `min_gap` typical spreads apart)."""
    S, O, out = sign.upper(), other.upper(), []
    for k, (yours_words, theirs_words) in FEATURE_WORDS.items():
        rt, ro, x = prof_target.get(k), prof_other.get(k), att.get(k)
        if not rt or not ro or x is None:
            continue
        sep = abs(rt["p50"] - ro["p50"]) / ((_spread(rt) + _spread(ro)) / 2)
        if sep < min_gap:
            continue
        dt, do = abs(x - rt["p50"]) / _spread(rt), abs(x - ro["p50"]) / _spread(ro)
        if do < dt * 0.6:
            part = "movement" if k in ("dir_x", "dir_y", "strokes", "travel") else "location" if k.startswith("loc") else "handshape"
            out.append({"part": part, "feature": k, "strength": round(sep * (dt - do), 3),
                        "say": f"Your hand {yours_words} like {O}; {S} {_contrast_words(k, rt, ro)}."})
    out.sort(key=lambda f: -f["strength"])
    return out[:max_points]


def _contrast_words(k: str, rt: dict, ro: dict) -> str:
    if k == "dir_x":
        return "slides more side to side" if rt["p50"] > ro["p50"] else "moves less side to side"
    if k == "dir_y":
        return "moves more up and down" if rt["p50"] > ro["p50"] else "moves less up and down, more along"
    if k == "strokes":
        return "repeats it more" if rt["p50"] > ro["p50"] else "does it fewer times"
    if k == "travel":
        return "has a bigger movement" if rt["p50"] > ro["p50"] else "has a smaller movement"
    if k == "loc_y":
        return "is signed lower" if rt["p50"] > ro["p50"] else "is signed higher"
    if k == "loc_x":
        return "is signed further out" if rt["p50"] > ro["p50"] else "is signed more toward the middle"
    if k.startswith("finger_") or k == "thumb_out":
        return "keeps it straighter" if rt["p50"] > ro["p50"] else "keeps it more bent"
    return "is different"
