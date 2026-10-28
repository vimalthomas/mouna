"""Per-part normalization for MediaPipe pose sequences — body frame + separate
local hand frames, with missing-keypoint imputation.

This replaces our earlier single-transform schemes (per-sequence z-score,
and the OpenHands clip-level shoulder-width scheme), neither of which match
what the sign-language literature actually does. Every competitive published
system (SPOTER, De Coster et al., Roh et al.) normalizes the body and each
hand SEPARATELY — collapsing both into one shared transform compresses hand
shape (the dominant signal for sign recognition) into a tiny slice of the
feature's dynamic range. Published gains from this: +8.14pp (De Coster),
+13.54pp (SignBart), +6.05pp (Roh et al.) on directly comparable tasks.

Formulation follows De Coster et al. (arXiv 2306.17558): body centered on
chest (mid-shoulder), scaled by shoulder distance; each hand centered on its
own wrist, scaled by wrist-to-middle-knuckle distance — this preserves
relative hand size (unlike SPOTER's per-hand min-max box, which discards it).

Missing-keypoint handling: MediaPipe leaves hands fully undetected in ~20-29%
of frames generally, and ~54% of frames on WLASL specifically (Roh et al.,
Table 1) — zero-filling these and then dividing by a near-zero empirical std
(our old per-sequence z-score) is what produced the 9000x outliers found
earlier today. Here we linearly interpolate missing frames from neighboring
detected frames first, so normalization never divides by a near-empty sample.
"""

import numpy as np

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9  # MediaPipe Hands: landmark 9 = middle finger MCP


def _interpolate_missing(block: np.ndarray) -> np.ndarray:
    """block: (T, N, 3). A landmark at frame t is "missing" if all 3 coords
    are exactly 0 (MediaPipe's own convention for undetected). Linearly
    interpolate each landmark's trajectory across missing frames using its
    own nearest valid frames; edge-gaps are filled from the nearest valid
    frame. A landmark missing in EVERY frame is left at zero — there's
    nothing to interpolate from."""
    T, N, C = block.shape
    out = block.copy()
    for j in range(N):
        valid = np.any(block[:, j, :] != 0, axis=-1)
        if not valid.any() or valid.all():
            continue
        valid_t = np.where(valid)[0]
        for c in range(C):
            out[:, j, c] = np.interp(np.arange(T), valid_t, block[valid_t, j, c])
    return out


def normalize_per_part(flat_225: np.ndarray) -> np.ndarray:
    """flat_225: (T, 225) raw MediaPipe coords (33 pose + 21 left + 21 right
    hand, x/y/z, NOT previously normalized). Returns (T, 225) normalized."""
    T = flat_225.shape[0]
    arr = flat_225.reshape(T, 75, 3).astype(np.float32)
    pose, lh, rh = arr[:, :33].copy(), arr[:, 33:54].copy(), arr[:, 54:75].copy()

    pose = _interpolate_missing(pose)
    lh = _interpolate_missing(lh)
    rh = _interpolate_missing(rh)

    # Body: per-frame center (removes signer drift), clip-level mean scale
    # (avoids amplifying frame-to-frame shoulder-estimate jitter into scale).
    mid_shoulder = (pose[:, LEFT_SHOULDER] + pose[:, RIGHT_SHOULDER]) / 2.0
    shoulder_dist = np.linalg.norm(pose[:, LEFT_SHOULDER] - pose[:, RIGHT_SHOULDER], axis=-1)
    valid_sd = shoulder_dist[shoulder_dist > 1e-6]
    body_scale = float(np.mean(valid_sd)) if len(valid_sd) else 1.0
    if body_scale < 1e-6:
        body_scale = 1.0
    pose_norm = (pose - mid_shoulder[:, None, :]) / body_scale

    def normalize_hand(hand: np.ndarray) -> np.ndarray:
        wrist = hand[:, WRIST_IDX]
        mcp = hand[:, MIDDLE_MCP_IDX]
        dist = np.linalg.norm(mcp - wrist, axis=-1)
        valid_d = dist[dist > 1e-6]
        scale = float(np.mean(valid_d)) if len(valid_d) else 1.0
        if scale < 1e-6:
            scale = 1.0
        return (hand - wrist[:, None, :]) / scale

    lh_norm = normalize_hand(lh)
    rh_norm = normalize_hand(rh)

    out = np.concatenate([pose_norm, lh_norm, rh_norm], axis=1)
    return out.reshape(T, 225).astype(np.float32)
