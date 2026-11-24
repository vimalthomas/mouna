"""LLM judge over the Score tab's evidence plus key frames.

Tested on the judgement benchmark by scripts/llm_evidence_judge.py (the
"vision" arm, 280 held-out attempts, 60 correct). Only hand count passed the
reliability bar there, beating our own hand-count check on every measure:

    LLM (measurements + frames): catches 76.2%, false alarm on correct 5.0%,
                                 false alarm on other signs with the same hand count 20.4%
    our hand-count check:        72.4% / 23.3% / 31.2%

So the app uses this judge for hand count only. The prompt and image layout
here are exactly the ones tested; change them and the numbers above no longer
apply until the benchmark is rerun.

Image: ROW 1 and ROW 2 are two fluent signers of the sign (pre-rendered by
scripts/build_llm_reference_rows.py into data/references/llm_refs/<sign>/),
ROW 3 is the attempt: three moments while the hands are up, cropped to face
and hands, each row 300 px high.
"""
from pathlib import Path

import cv2
import numpy as np

REF_DIR = Path(__file__).resolve().parents[3] / "data/references/llm_refs"
ROW_HEIGHT = 300
PARTS = ("handshape", "orientation", "location", "hands")

# benchmark result for the hand-count use (scripts/llm_evidence_judge.py, arm "vision")
HAND_COUNT_RELIABILITY = {"catches": 0.762, "false_alarm": 0.05, "false_alarm_other_signs": 0.204, "n_correct": 60}

CP_WORDS = {"palm_x": "palm facing sideways", "palm_y": "palm facing up/down", "palm_z": "palm facing toward/away",
            "finger_thumb": "thumb bend", "finger_index": "index finger bend", "finger_middle": "middle finger bend",
            "finger_ring": "ring finger bend", "finger_little": "little finger bend", "thumb_out": "thumb position",
            "loc_x": "hand side-to-side position", "loc_y": "hand height", "dir_x": "side-to-side motion",
            "dir_y": "up-down motion", "strokes": "number of strokes", "travel": "size of movement", "gap": "hands touching"}

PROMPT = """You judge an attempt at the American Sign Language sign "{sign}".
Measurements compare the attempt with fluent signers of "{sign}" (not with one video).
Each parameter has a level (fluent / close / needs_work), a 0-100 score, the
measurements furthest outside the fluent range, and how trustworthy that score
is on held-out data. Scores marked "estimate" are hints and can be wrong;
tracking noise also produces small deviations in correct signing.

{evidence}
{image_note}
Decide, per part, whether the attempt's part is DIFFERENT from how fluent signers
do "{sign}" in a way that changes the sign:
- handshape: fingers extended/bent, thumb position
- orientation: which way the palm faces
- location: where the hand is relative to face and body
- hands: one or two hands used
Say "different" only with clear support; "same" otherwise; "unsure" if the evidence is missing.
Reply with JSON only:
{{"handshape": "...", "orientation": "...", "location": "...", "hands": "...",
  "how": "<for any 'different': what the attempt does instead, in a few words; else empty>"}}"""

IMAGE_NOTE = """An image is attached. ROW 1 and ROW 2: two fluent signers doing "{sign}"; their
differences are normal variation. ROW 3: the attempt. Use it together with the measurements."""


def evidence_text(ps: dict, sim: float | None, hands_match, rel: dict) -> str:
    lines = [f"Whole-sign similarity to fluent signers: {sim:.2f} (good >= 0.60)" if sim == sim and sim is not None else ""]
    for p in ("handshape", "location", "orientation", "movement"):
        e = ps.get(p) or {}
        if not e.get("level"):
            lines.append(f"- {e.get('label', p)}: not measured")
            continue
        trust = "trusted" if e.get("reliable") else "estimate"
        r = rel.get(p)
        if r:
            trust += f"; on held-out signs it flags {r['test_false_alarm']:.0%} of correct signing and catches {r['test_catches']:.0%} of real differences"
        s = f"- {e['label']}: {e['level']} (score {e['score']}; {trust})"
        ev = e.get("evidence") or []
        if ev:
            s += "; furthest outside: " + "; ".join(
                f"{x['dim']} {x['yours']}{'°' if x['unit'] == '°' else ''} vs fluent {x['fluent'][0]}-{x['fluent'][1]}{'°' if x['unit'] == '°' else ' shoulder widths'}"
                for x in ev[:3])
        elif e.get("cp_worst") and e["level"] != "fluent":
            s += f"; worst measurement: {CP_WORDS.get(e['cp_worst'], e['cp_worst'])}"
        lines.append(s)
    if hands_match is not None:
        lines.append(f"- Hand count: {'matches' if hands_match else 'differs from'} what fluent signers use (an estimate; flags about 22% of correct signing)")
    lines.append("- Facial expression: not measured")
    return "\n".join(x for x in lines if x)


def prompt(sign: str, evidence: str, with_image: bool = True) -> str:
    return PROMPT.format(sign=sign, evidence=evidence, image_note=IMAGE_NOTE.format(sign=sign) if with_image else "")


def parse_answer(d) -> dict:
    if isinstance(d, list):   # some replies wrap the object in a list
        d = next((x for x in d if isinstance(x, dict)), {})
    return {**{k: str(d.get(k, "unsure")).lower() for k in PARTS}, "note": d.get("how", "") or ""}


# ---- image -------------------------------------------------------------------------------

def pick_frames(kp: np.ndarray, n_frames: int) -> tuple[list[int], tuple[float, float, float, float] | None]:
    """Which 3 frames to show and the crop box (fractions of the frame), from
    (T, 75, 3) keypoints aligned with the frames: moments at 25/50/75% of the
    stretch where the hands are visible, cropped to face + hands (as
    llm_vision_judge.key_frames)."""
    idx = [int(n_frames * a) for a in (0.3, 0.5, 0.7)]
    box = None
    if kp is not None and len(kp):
        hands = np.any(kp[:, 33:75] != 0, axis=(1, 2))
        up = np.where(hands)[0]
        if len(up) >= 3:
            lo, hi = up[0], up[-1]
            idx = [int((lo + (hi - lo) * a) * n_frames / len(kp)) for a in (0.25, 0.5, 0.75)]
        pts = kp[:, [0, 2, 5, 7, 8, 11, 12, 15, 16] + list(range(33, 75)), :2].reshape(-1, 2)
        pts = pts[np.all(pts > 0, axis=1) & np.all(pts < 1, axis=1)]
        if len(pts) > 10:
            x0, y0 = np.percentile(pts, 2, axis=0)
            x1, y1 = np.percentile(pts, 98, axis=0)
            box = (float(x0), float(y0), float(x1), float(y1))
    return [max(0, min(n_frames - 1, i)) for i in idx], box


def crop_row(frames: list[np.ndarray], box, height: int = ROW_HEIGHT) -> np.ndarray | None:
    out = []
    for f in frames:
        if f is None:
            continue
        H, W = f.shape[:2]
        if box:
            x0, y0, x1, y1 = box
            cx, cy, half = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H, max((x1 - x0) * W, (y1 - y0) * H) * 0.62
            f = f[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
        if f.size:
            out.append(cv2.resize(f, (int(f.shape[1] * height / max(1, f.shape[0])), height)))
    return np.hstack(out[:3]) if len(out) >= 2 else None


def stack_rows(rows: list[np.ndarray]) -> bytes | None:
    """Rows padded to one width, 10 px white gaps, JPEG (as llm_vision_judge.sheet)."""
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    gap = np.full((10, w, 3), 255, np.uint8)
    stacked = [x for r in rows for x in (r, gap)][:-1]
    ok, buf = cv2.imencode(".jpg", np.vstack(stacked), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


def reference_rows(sign: str) -> list[np.ndarray] | None:
    d = REF_DIR / sign.lower()
    rows = [cv2.imread(str(d / f"row{i}.jpg")) for i in (1, 2)]
    return rows if all(r is not None for r in rows) else None


def has_references(sign: str) -> bool:
    return all((REF_DIR / sign.lower() / f"row{i}.jpg").exists() for i in (1, 2))
