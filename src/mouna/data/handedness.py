"""Canonicalize signer handedness so the model sees one consistent
left/right convention instead of a bimodal mix it can't generalize across.

Evidence this matters: QA review of accident/africa/all surfaced 14+ clips
independently flagged "mirror hand" / "different hand" -- all scoring
near-zero model confidence (0.001-0.07) despite being confirmed correct
signs. The model has essentially learned one handedness and fails on the
other.

Operates on RAW (pre-normalize_per_part) flat keypoints, in MediaPipe's
native [0,1] image-coordinate space, where a true left-right mirror is
x' = 1 - x plus swapping every left/right landmark pair -- not just the
hand blocks. (The existing training-time flip augmentation in
train_transformer.py uses x' = 1-x against ALREADY per-part-normalized,
centered data where that formula is wrong -- see MOUNA_FIX_FLIP_AUG.)
"""

import numpy as np

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

# MediaPipe Pose left/right landmark pairs that must be swapped (not just
# x-flipped in place) for an anatomically correct mirror.
POSE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),  # eyes, ears, mouth corners
    (11, 12), (13, 14), (15, 16),              # shoulder, elbow, wrist
    (17, 18), (19, 20), (21, 22),              # pinky, index, thumb (hand-adjacent pose pts)
    (23, 24), (25, 26), (27, 28),              # hip, knee, ankle
    (29, 30), (31, 32),                        # heel, foot_index
]

MIN_ACTIVITY_RATIO = 0.6  # below this, treat the weaker hand as "not really active"


def _hand_activity(hand: np.ndarray) -> float:
    """hand: (T, 21, 3). Motion energy (frame-to-frame displacement) summed
    over frames where the hand was actually detected -- a hand that's
    present but static still counts via its presence; one that's barely
    detected contributes almost nothing."""
    present = np.any(hand != 0, axis=(1, 2))
    if present.sum() < 2:
        return 0.0
    diffs = np.diff(hand[present], axis=0)
    return float(np.linalg.norm(diffs, axis=-1).sum()) + float(present.sum())


def detect_dominant_hand(flat_225_raw: np.ndarray) -> str:
    """Returns 'left', 'right', 'both' (symmetric two-handed sign, e.g.
    accident's fist-collide -- mirroring would be meaningless), or
    'neither' (no hand detected at all)."""
    arr = flat_225_raw.reshape(-1, 75, 3)
    lh, rh = arr[:, 33:54], arr[:, 54:75]
    l_act, r_act = _hand_activity(lh), _hand_activity(rh)
    if l_act == 0 and r_act == 0:
        return "neither"
    if l_act == 0:
        return "right"
    if r_act == 0:
        return "left"
    ratio = min(l_act, r_act) / max(l_act, r_act)
    if ratio >= MIN_ACTIVITY_RATIO:
        return "both"
    return "left" if l_act > r_act else "right"


def mirror_raw_keypoints(flat_225_raw: np.ndarray) -> np.ndarray:
    """True left-right mirror of raw [0,1]-space MediaPipe keypoints:
    flips x everywhere, swaps left_hand<->right_hand blocks, and swaps
    every left/right pose landmark pair (not just negates their x in
    place -- a flipped image moves the anatomical left shoulder to where
    the right shoulder was)."""
    T = flat_225_raw.shape[0]
    arr = flat_225_raw.reshape(T, 75, 3).copy()

    nonzero = np.any(arr != 0, axis=-1)
    arr[..., 0] = np.where(nonzero, 1.0 - arr[..., 0], arr[..., 0])

    pose = arr[:, :33].copy()
    for a, b in POSE_LR_PAIRS:
        pose[:, a], pose[:, b] = arr[:, b].copy(), arr[:, a].copy()
    arr[:, :33] = pose

    lh, rh = arr[:, 33:54].copy(), arr[:, 54:75].copy()
    arr[:, 33:54], arr[:, 54:75] = rh, lh

    return arr.reshape(T, 225).astype(np.float32)


def canonicalize_handedness(flat_225_raw: np.ndarray, target: str = "right") -> np.ndarray:
    """Mirror the clip only if its dominant hand isn't `target`. Symmetric
    two-handed signs ('both') and undetected clips ('neither') pass
    through unchanged -- there's nothing to canonicalize."""
    dominant = detect_dominant_hand(flat_225_raw)
    if dominant == target or dominant in ("both", "neither"):
        return flat_225_raw
    return mirror_raw_keypoints(flat_225_raw)
