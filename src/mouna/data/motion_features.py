"""Augment the 225-dim per-part-normalized pose features with two explicit
signals the raw landmark positions only encode weakly:

  - Palm orientation (6 dims: 3 per hand) — the unit normal of the plane
    formed by wrist + index-MCP + pinky-MCP. Diagnosed directly from our own
    data: `mother` vs `brown` and `accident` vs `play` have near-zero
    between-class separation in raw hand landmarks relative to within-class
    noise, while well-classified pairs (walk/computer) separate clearly on
    hands. Orientation is part of ASL's real phonology (handshape, location,
    movement, ORIENTATION, non-manual) and isn't explicitly present in raw
    x/y/z — the model has to infer which way the palm faces from the
    scatter of finger joints, a noisy indirect signal.

  - Frame-to-frame velocity (225 dims — same shape as position) — makes
    motion PATTERN (e.g. `mother`'s repeated tap vs `brown`'s continuous
    slide) an explicit input instead of something the transformer must
    derive from position differences across attention steps.

Output: 225 (position) + 225 (velocity) + 6 (orientation) = 456 dims.
"""

import numpy as np

WRIST_IDX = 0
INDEX_MCP_IDX = 5
PINKY_MCP_IDX = 17


def _hand_orientation(hand: np.ndarray) -> np.ndarray:
    """hand: (T, 21, 3). Returns (T, 3) unit normal per frame."""
    wrist = hand[:, WRIST_IDX]
    index_mcp = hand[:, INDEX_MCP_IDX]
    pinky_mcp = hand[:, PINKY_MCP_IDX]
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)  # avoid div-by-zero when hand undetected (all zero)
    return normal / norm


def add_motion_features(flat_225: np.ndarray) -> np.ndarray:
    """flat_225: (T, 225) per-part-normalized pose features.
    Returns (T, 456): [position(225), velocity(225), orientation(6)]."""
    T = flat_225.shape[0]
    arr = flat_225.reshape(T, 75, 3)
    lh, rh = arr[:, 33:54], arr[:, 54:75]

    velocity = np.diff(flat_225, axis=0, prepend=flat_225[:1])  # (T, 225), first frame = 0

    lh_orient = _hand_orientation(lh)  # (T, 3)
    rh_orient = _hand_orientation(rh)  # (T, 3)
    orientation = np.concatenate([lh_orient, rh_orient], axis=1)  # (T, 6)

    return np.concatenate([flat_225, velocity, orientation], axis=1).astype(np.float32)  # (T, 456)
