"""Measured movement clues: the learner's motion vs the target's.

Movement isn't graded yet (no movement check has passed the judgement
benchmark), but three things can be measured and told as hints:
  - contact: for two-handed signs where ASL-LEX codes contact (PAPER,
    SCHOOL), did the hands come together?
  - strokes: how many back-and-forth strokes (once vs repeated)?
  - travel: how far did the dominant hand move, relative to shoulder width?
Each is compared with the target's reference clip; only clear differences
become hints. Both inputs are canonical keypoints (dominant hand = right slot).
"""
import numpy as np

RIGHT, LEFT = slice(54, 75), slice(33, 54)


def _norm(seq: np.ndarray):
    """(T,75,3) -> wrist path of the dominant hand, both hands' centres,
    hand size and shoulder width -- in shoulder-width units."""
    a = np.asarray(seq, dtype=np.float32).reshape(len(seq), 75, 3)[:, :, :2]
    sh = a[:, [11, 12]]
    ok = np.all(sh != 0, axis=(1, 2))
    if ok.sum() < 3:
        return None
    sw = float(np.median(np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1))) or 1.0
    mid = np.median((sh[ok, 0] + sh[ok, 1]) / 2, axis=0)
    r, l = a[:, RIGHT], a[:, LEFT]
    rp, lp = np.any(r != 0, axis=(1, 2)), np.any(l != 0, axis=(1, 2))
    size = float(np.median(np.linalg.norm(r[rp, 9] - r[rp, 0], axis=1))) if rp.sum() else 0.0
    return {"wrist": (r[rp, 0] - mid) / sw, "rc": r.mean(1), "lc": l.mean(1), "rp": rp, "lp": lp,
            "size": size, "sw": sw}


def _travel(w: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0


def _strokes(w: np.ndarray, min_amp=0.08) -> int:
    """Back-and-forth strokes along the main direction of motion."""
    if len(w) < 4:
        return 0
    c = w - w.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    x = c @ axis
    # turning points with at least min_amp (shoulder widths) between them
    turns, last = 0, x[0]
    direction = 0
    for v in x[1:]:
        d = v - last
        if abs(d) < min_amp:
            continue
        s = 1 if d > 0 else -1
        if direction and s != direction:
            turns += 1
        direction, last = s, v
    return turns // 2 + 1


def _min_gap(n: dict) -> float | None:
    both = n["rp"] & n["lp"]
    if both.sum() < 2 or not n["size"]:
        return None
    return float(np.min(np.linalg.norm(n["rc"][both] - n["lc"][both], axis=1)) / n["size"])


def movement_hints(attempt_canon: np.ndarray, reference_canon: np.ndarray, sign: str,
                   two_handed_contact: bool = False) -> tuple[list[str], dict]:
    """Plain hints (only clear differences) and the measured numbers."""
    a, r = _norm(attempt_canon), _norm(reference_canon)
    if a is None or r is None or len(a["wrist"]) < 3 or len(r["wrist"]) < 3:
        return [], {}
    S = sign.upper()
    nums = {"travel": (_travel(a["wrist"]), _travel(r["wrist"])),
            "strokes": (_strokes(a["wrist"]), _strokes(r["wrist"])),
            "gap": (_min_gap(a), _min_gap(r))}
    hints = []
    ga, gr = nums["gap"]
    if two_handed_contact and gr is not None and gr < 1.0:
        if ga is None:
            hints.append(f"Only one hand was seen; in {S} one hand touches the other.")
        elif ga > 1.6:
            hints.append(f"Your hands didn't come together; in {S} one hand touches the other.")
    sa, sr = nums["strokes"]
    if sr >= 2 and sa == 1:
        hints.append(f"In {S} the movement repeats ({sr} strokes); you did it once.")
    elif sr == 1 and sa >= 3:
        hints.append(f"In {S} the movement happens once; you repeated it.")
    ta, tr = nums["travel"]
    if tr > 0.15:
        if ta < 0.45 * tr:
            hints.append(f"Your hand moved much less than in the example; make the {S} movement bigger.")
        elif ta > 2.2 * tr:
            hints.append(f"Your hand moved much more than in the example; keep the {S} movement smaller.")
    return hints, {k: [None if v is None else round(v, 3) for v in vals] for k, vals in nums.items()}
