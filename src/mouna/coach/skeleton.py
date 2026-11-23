"""Normalized skeleton sequences for display -- renders BOTH hands and the
FULL pose (not just the dominant hand/one arm, which was a real bug: it
made every figure look one-sided, and made an attempt's single rendered
hand visually overlap the reference's single rendered hand in a way a
real user correctly read as "my hands are in the same place").

Renders in the SAME body-relative coordinate frame (centered on
mid_shoulder, scaled by shoulder width) so two different people on two
different cameras are actually comparable, not just two unrelated
raw-pixel traces overlaid on top of each other.

2D only (x,y), deliberately -- see parameters.py's location_traj comment:
hand-model z and pose-model z aren't on a comparable scale, so a 3D
skeleton mixing both would be visually misleading in exactly the way that
bug was. Each hand's palm-orientation arrow is computed from that hand's
own local vectors only (self-consistent, no cross-model mixing), then
projected to 2D for display.

This function is agnostic to RAW vs. CANONICALIZED input -- callers pass
either the raw (as detected, correctly-sided) buffer or the
canonicalize_handedness'd buffer (the space scoring actually happens in,
possibly mirrored). Rendering both of a live attempt, side by side, lets a
user see their actual motion AND what the system compares against.
"""

from typing import Optional

import numpy as np

from mouna.coach import face_mesh
from mouna.data.part_normalize import _interpolate_missing  # reuse validated gap-filling, not reimplement it

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
WRIST = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
PINKY_MCP = 17

# Full upper-body chain -- both arms, not just the dominant-hand side.
# Hips included so the frontend can draw a filled torso quad instead of a
# bare shoulder-to-shoulder line -- the pose model already gives us these
# for free, just wasn't being read before.
POSE_RENDER_INDICES = {
    "nose": 0, "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
}


def _body_frame(pose_raw: np.ndarray):
    pose = _interpolate_missing(pose_raw)
    mid_shoulder = (pose[:, LEFT_SHOULDER] + pose[:, RIGHT_SHOULDER]) / 2.0
    shoulder_dist = np.linalg.norm(pose[:, LEFT_SHOULDER] - pose[:, RIGHT_SHOULDER], axis=-1)
    valid = shoulder_dist[shoulder_dist > 1e-6]
    scale = float(np.mean(valid)) if len(valid) else 1.0
    return pose, mid_shoulder, (scale if scale > 1e-6 else 1.0)


def _normalize_hand_shape(hand_raw: np.ndarray) -> np.ndarray:
    """Wrist-centered, scaled by wrist-to-middle-MCP distance -- same
    formula as parameters.py's _normalize_hand, needed here only for the
    orientation vector (cross product wants a consistent local scale)."""
    hand = _interpolate_missing(hand_raw)
    wrist = hand[:, WRIST]
    mcp = hand[:, MIDDLE_MCP]
    dist = np.linalg.norm(mcp - wrist, axis=-1)
    valid = dist[dist > 1e-6]
    scale = float(np.mean(valid)) if len(valid) else 1.0
    if scale < 1e-6:
        scale = 1.0
    return (hand - wrist[:, None, :]) / scale


def _orientation_2d(hand_shape: np.ndarray) -> np.ndarray:
    v1, v2 = hand_shape[:, INDEX_MCP], hand_shape[:, PINKY_MCP]
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm_len = np.where(norm_len < 1e-6, 1.0, norm_len)
    return (normal / norm_len)[:, :2]


def extract_skeleton_sequence(
    flat_225: np.ndarray,
    n_frames: int = 20,
    frame_indices: Optional[list[int]] = None,
    face_frames: Optional[list[Optional[dict]]] = None,
    mirror_face: bool = False,
) -> Optional[list[dict]]:
    """flat_225: (T, 225) keypoints, RAW or canonicalized (caller's choice).
    Returns up to n_frames evenly-sampled frames (only frames where at
    least one hand was detected), each:
      {"pose": {name: [x,y], ...}, "left_hand": [[x,y],x21] | None,
       "right_hand": [[x,y],x21] | None, "left_orientation": [ox,oy] | None,
       "right_orientation": [ox,oy] | None, "face": {region: [[x,y],...]} | None}
    all in body-relative (mid_shoulder-centered, shoulder_dist-scaled)
    space. Pass `frame_indices` to get specific original-array frames
    instead of an evenly-sampled sequence (used for snapshot moments).
    None if neither hand was ever detected.

    face_frames: same-length, same-index-aligned list as the raw buffer
    this flat_225 came from (keypoint_service.extract_face_from_base64()
    output per frame, or the reference's per-frame landmarks_xy list) --
    omit for no face rendering. mirror_face: set True when flat_225 is the
    MIRRORED (canonicalize_handedness'd) version of the signer's own raw
    data -- face landmarks live in a completely separate array from the
    225-dim pose/hand one, so canonicalize_handedness never touches them;
    this flag is how the caller tells us to apply the same x'=1-x mirror
    here, so a mirrored body doesn't end up paired with an un-mirrored
    face in the canonical view."""
    T = flat_225.shape[0]
    arr = flat_225.reshape(T, 75, 3).astype(np.float32)
    pose_raw, lh_raw, rh_raw = arr[:, :33], arr[:, 33:54], arr[:, 54:75]

    left_present = np.any(lh_raw != 0, axis=(1, 2))
    right_present = np.any(rh_raw != 0, axis=(1, 2))
    any_present = left_present | right_present
    valid_idx = np.where(any_present)[0]
    if len(valid_idx) < 2:
        return None

    pose, mid_shoulder, scale = _body_frame(pose_raw)
    lh = _interpolate_missing(lh_raw)
    rh = _interpolate_missing(rh_raw)
    lh_shape = _normalize_hand_shape(lh_raw)
    rh_shape = _normalize_hand_shape(rh_raw)

    pose_norm = (pose[:, :, :2] - mid_shoulder[:, None, :2]) / scale  # (T, 33, 2)
    lh_norm = (lh[:, :, :2] - mid_shoulder[:, None, :2]) / scale      # (T, 21, 2)
    rh_norm = (rh[:, :, :2] - mid_shoulder[:, None, :2]) / scale      # (T, 21, 2)

    lh_orient = _orientation_2d(lh_shape)
    rh_orient = _orientation_2d(rh_shape)

    if frame_indices is not None:
        sample_idx = [i for i in frame_indices if 0 <= i < T]
    else:
        sample_idx = valid_idx[np.linspace(0, len(valid_idx) - 1, min(n_frames, len(valid_idx))).astype(int)]

    frames = []
    denom = max(T - 1, 1)
    for i in sample_idx:
        frames.append({
            "pose": {name: pose_norm[i, idx].tolist() for name, idx in POSE_RENDER_INDICES.items()},
            "left_hand": lh_norm[i].tolist() if left_present[i] else None,
            "right_hand": rh_norm[i].tolist() if right_present[i] else None,
            "left_orientation": lh_orient[i].tolist() if left_present[i] else None,
            "right_orientation": rh_orient[i].tolist() if right_present[i] else None,
            "face": _face_contours_at(face_frames, i, mid_shoulder, scale, mirror_face),
            # Position of this frame within the whole buffer, 0..1 -- lets
            # the frontend sync a frame to the recorded attempt video.
            "t": float(i) / denom,
            # Inverse of the body-relative normalization, in the source
            # image's normalized [0,1] coords: image_xy = xy * scale + origin.
            # Used to draw skeletons back onto the signer's own video.
            "origin": mid_shoulder[i, :2].tolist(),
            "scale": scale,
            # Hand-local 3D (wrist-centered, wrist->middle-MCP scaled). Hand
            # z is self-consistent within one hand model, unlike hand-vs-pose
            # z (see module docstring), so it's safe for a hands-only 3D view.
            "left_hand_3d": lh_shape[i].tolist() if left_present[i] else None,
            "right_hand_3d": rh_shape[i].tolist() if right_present[i] else None,
        })
    return frames


def _face_contours_at(
    face_frames: Optional[list], i: int, mid_shoulder: np.ndarray, scale: float, mirror: bool,
) -> Optional[dict]:
    """face_frames elements come in two shapes depending on caller: the live
    path's face_buffer is a list of dicts (keypoint_service.
    extract_face_from_base64() output, {"landmarks_xy": ..., ...} or None);
    the reference path's landmarks_xy_sequence is the bare per-frame point
    list (or None) directly, already pre-extracted at build time (see
    auto_screen_references.py). Handle both rather than forcing one shape
    on the other caller."""
    if not face_frames or i >= len(face_frames) or face_frames[i] is None:
        return None
    entry = face_frames[i]
    landmarks_xy = entry.get("landmarks_xy") if isinstance(entry, dict) else entry
    if landmarks_xy is None:
        return None
    pts = np.asarray(landmarks_xy, dtype=np.float32)
    if mirror:
        pts = pts.copy()
        pts[:, 0] = 1.0 - pts[:, 0]
    contours = face_mesh.extract_face_contours(pts, mid_shoulder[i, :2], scale)
    return contours or None
