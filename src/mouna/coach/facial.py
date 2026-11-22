"""Facial non-manual-marker (NMM) checks: brow furrow (for wh-question-marked
signs) and headshake (for negation-marked signs).

VALIDATED (2026-11-21, scripts/validate_nms_thresholds.py) against held-out
clips -- not just the 1 representative clip per word the thresholds were
originally eyeballed against. Positive set: every auto-screened what/who/how
clip (n=105) for brow furrow, every auto-screened no clip (n=35) for
headshake. Negative set: a random sample of 12 non-NMS words, 10 clips each
(n=120), to measure false positives. Results, honestly:

  brow furrow (wh_question_marker), threshold=0.3:
    true positive rate (recall) 53.3%, false positive rate 27.5%.
    Youden's-J-optimal threshold is 0.098 (J=0.298 vs current J=0.258) --
    barely better, and only by trading most of the false-positive gain for
    recall at the very low end of the range. The real finding is NOT "the
    threshold was wrong" -- it's that peak brow-furrow alone is a real but
    WEAK, noisy signal (no single cutpoint cleanly separates positive from
    negative). Kept at 0.3, favoring precision over recall, as a deliberate
    (if debatable) choice, not a default -- see git history for the
    distribution means if revisiting this.

  headshake (negation_marker), threshold=8.0 degrees:
    true positive rate 71.4%, false positive rate 34.2%. Youden's-J-optimal
    threshold is 7.03 degrees (J=0.383 vs current J=0.373) -- current
    threshold was already near-optimal for this one-feature split; this
    is NOT a miscalibration, the signal itself still has real overlap
    between positive and negative (natural head movement during ANY sign
    crosses 7-8 degrees of yaw often enough to cost a third false-positive
    rate here).

Conclusion for both: single-scalar peak/range features are real, validated,
better-than-chance signals, but neither is reliable enough on its own to
present as a solved NMS detector -- report results as informational with
this caveat, not as validated the way handshape/location/orientation are
(see parameters.py). A genuine improvement would need a richer feature
(sustained-duration not just peak, multi-blendshape combination, or
per-signer neutral-face baselining) or the same discrete-bucket-ensemble
approach that rescued the geometric parameters after continuous distances
failed -- not a different single threshold on the same single feature.

Reference side: scripts/auto_screen_references.py stores
references[word]["representative_face_signal"] = {
    "blendshape_names": [...52 names...],
    "blendshape_matrix": (T, 52) float array,
    "head_yaw_deg": (T,) float array,
}
Live side: see from_live_frames() below, which converts
keypoint_service.extract_face_from_base64()'s per-frame output into the
same shape.
"""

from typing import Optional

import numpy as np

# WH-questions (who/what/where/when/why/how) are marked by FURROWED (lowered)
# brows, not raised ones -- raised brows mark yes/no questions instead, a
# different grammatical signal. Corrected from an initial wrong guess after
# checking real reference clips: browInnerUp/browOuterUp* sat near zero for
# what/who/how, while browDownLeft/Right showed a real signal (what: mean
# 0.40, max 0.84; who: mean 0.18, max 0.37 -- see git history for the
# before/after numbers). Still only 3 reference words' worth of evidence,
# still not a real within/across calibration -- see module docstring.
BROW_FURROW_NAMES = ["browDownLeft", "browDownRight"]
BROW_FURROW_THRESHOLD = 0.3  # ARKit blendshape score, 0-1 -- fits what/who/how's observed max, not guessed blind
HEADSHAKE_YAW_RANGE_DEG = 8.0  # minimum yaw oscillation to call it a headshake -- reasoned default, not calibrated


def from_live_frames(frames: list[Optional[dict]]) -> Optional[dict]:
    """frames: list of keypoint_service.extract_face_from_base64() outputs
    (each {"blendshapes": {...}, "head_yaw_deg": float|None} or None for a
    frame where no face was detected). Returns the same shape
    auto_screen_references.py produces for the reference side, or None if
    no frame in the buffer had a detected face."""
    valid = [f for f in frames if f is not None]
    if not valid:
        return None
    names = list(valid[0]["blendshapes"].keys())
    matrix = np.array([[f["blendshapes"].get(n, 0.0) for n in names] for f in valid])
    yaws = np.array([f["head_yaw_deg"] for f in valid if f["head_yaw_deg"] is not None], dtype=np.float32)
    return {"blendshape_names": names, "blendshape_matrix": matrix, "head_yaw_deg": yaws}


def _brow_furrow_score(face_signal: dict) -> float:
    """Peak furrow across the clip, not the mean -- the furrow is a held
    moment within the sign, not necessarily sustained the whole clip, so
    averaging it away would wash out exactly the signal we want."""
    names = face_signal["blendshape_names"]
    idxs = [names.index(n) for n in BROW_FURROW_NAMES if n in names]
    if not idxs:
        return 0.0
    return float(face_signal["blendshape_matrix"][:, idxs].max())


def _yaw_range(face_signal: dict) -> float:
    yaw = face_signal.get("head_yaw_deg")
    if yaw is None or len(yaw) < 2:
        return 0.0
    return float(yaw.max() - yaw.min())


def check_nms(face_signal: Optional[dict], nms_type: str) -> dict:
    """Returns {"present": bool, "detected": bool|None, "method": str}.
    `present=False` means no face was ever detected in the buffer at all
    (different from `detected=False`, which means a face WAS seen but the
    specific marker wasn't)."""
    if face_signal is None:
        return {"present": False, "detected": False, "method": nms_type}
    if nms_type == "wh_question_marker":
        return {"present": True, "detected": _brow_furrow_score(face_signal) >= BROW_FURROW_THRESHOLD, "method": nms_type}
    if nms_type == "negation_marker":
        return {"present": True, "detected": _yaw_range(face_signal) >= HEADSHAKE_YAW_RANGE_DEG, "method": nms_type}
    return {"present": True, "detected": None, "method": nms_type}
