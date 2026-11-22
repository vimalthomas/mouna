"""Face mesh contour groups for skeleton rendering -- which of the 478
FaceLandmarker points trace a recognizable face outline (jaw, eyes,
eyebrows, lips), plus the body-relative transform to put them in the same
coordinate space skeleton.py already renders pose/hands in.

Index lists pulled directly from the installed MediaPipe Tasks library
(`mediapipe.tasks.python.vision.FaceLandmarksConnections`), not guessed or
recalled -- that class ships the exact edge lists the legacy
`solutions.face_mesh` drawing utils used, which this MediaPipe version no
longer exposes directly. Each region's edges were chain-walked into an
ordered point loop/path (see git history for the extraction script) so the
frontend can draw a single <polyline>/<polygon> per region instead of a
scatter of unconnected segments.

Eyebrows come back from MediaPipe as TWO disjoint 5-point strips each (the
upper and lower edge of the brow band) -- merged into one 10-point outline
per eyebrow here, in upper-strip-then-reversed-lower-strip order, so it
draws as a single closed band shape instead of two separate floating lines.
Lips are two genuinely separate closed loops (outer lip ring, inner mouth
opening) and stay separate -- collapsing them would lose the open-mouth
signal.
"""

import numpy as np

FACE_OVAL = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
             365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338,
             10, 109, 67, 103, 54, 21, 162, 127, 234, 93]
LEFT_EYE = [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362, 398]
RIGHT_EYE = [160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159]
# Merge the two disjoint strips MediaPipe returns per eyebrow into one band.
LEFT_EYEBROW = [300, 293, 334, 296, 336] + [285, 295, 282, 283, 276][::-1]
RIGHT_EYEBROW = [70, 63, 105, 66, 107] + [55, 65, 52, 53, 46][::-1]
LIPS_OUTER = [0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267]
LIPS_INNER = [13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312]

REGIONS = {
    "face_oval": FACE_OVAL,
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "left_eyebrow": LEFT_EYEBROW,
    "right_eyebrow": RIGHT_EYEBROW,
    "lips_outer": LIPS_OUTER,
    "lips_inner": LIPS_INNER,
}


def extract_face_contours(landmarks_xy: np.ndarray, mid_shoulder_xy: np.ndarray, scale: float) -> dict:
    """landmarks_xy: (478, 2) raw image-plane (x,y) for ONE frame, in the
    SAME normalized image coordinates pose/hand landmarks come in (all three
    MediaPipe tasks run on the same source frame). mid_shoulder_xy/scale:
    from skeleton.py's _body_frame() for that same frame, so face points
    land in the identical body-relative space pose/hands already use --
    only x,y, matching the rest of this module's cross-model-z avoidance.

    Returns {region_name: [[x,y], ...]} or {} if landmarks_xy is None
    (face not detected that frame)."""
    if landmarks_xy is None:
        return {}
    pts = (np.asarray(landmarks_xy, dtype=np.float32) - mid_shoulder_xy) / scale
    return {name: pts[idxs].tolist() for name, idxs in REGIONS.items()}
