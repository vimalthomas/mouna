"""Facial non-manual-marker (NMM) checks: brow furrow (for wh-question-marked
signs) and headshake (for negation-marked signs).

UNVALIDATED, first pass -- unlike handshape/location/orientation (see
parameters.py), these have NOT been through a within-word-vs-across-word
calibration. That check needs a broad set of confirmed reference clips for
multiple NMS-tagged words to compare against each other, and right now we
only have the 4 words (what/who/how/no) this feature was built for, built
via an explicitly weaker "auto_screened" method (see
scripts/auto_screen_references.py), not human QA. The brow-furrow threshold
was at least checked against those 3 reference clips' real blendshape
values (see BROW_FURROW_THRESHOLD's comment) -- the headshake threshold is
still a reasoned guess, not checked against anything. Report results as
informational, with that caveat, not as reliable as the three validated
manual parameters.

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
