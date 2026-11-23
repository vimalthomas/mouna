"""3D skin data for a learner's own attempt (results screen, v3).

Same format as the example skins (scripts/build_sign_skin.py), built from
what the live session already has -- the 225-float keypoints per frame and
the face mesh per frame -- and cleaned with the same steps as the examples
(mouna.skin.pipeline): drop impossible jumps, fill short gaps, speed-aware
zero-lag smoothing, steady 3D finger lengths. Live frames come at ~10 fps,
so the smoothing window is shorter than for 25-30 fps example videos.

The learner sees themselves mirrored (like the camera preview), so x is
flipped for display.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from mouna.skin.pipeline import _tracks, fill_gaps, fix_bones, smooth, drop_jumps

WIDTH, HEIGHT, FPS = 640, 480, 10.0   # camera-view.component captures 640x480 every 100 ms
SETTINGS = {"jump": 0.8, "max_gap": 2, "sigma": 0.8, "fast": 0.3, "face_sigma": 0.7}


def _r(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), 1).tolist()


def attempt_skin(keypoints: list[list[float]], face_frames: list[dict | None] | None,
                 width: int = WIDTH, height: int = HEIGHT, fps: float = FPS, mirror: bool = True) -> dict | None:
    if not keypoints:
        return None
    px = np.array([width, height, width], dtype=np.float64)   # MediaPipe z uses x's scale
    flip = (lambda a: np.column_stack([width - a[:, 0], a[:, 1:]])) if mirror else (lambda a: a)  # noqa: E731
    arr = np.asarray(keypoints, dtype=np.float64).reshape(len(keypoints), 75, 3)

    # hands: the image points double as the 3D shape (z is MediaPipe's relative depth)
    hands_by_frame = []
    for f in arr:
        hs = []
        for block in (f[33:54], f[54:75]):
            if np.any(block != 0):
                img = flip(block * px)
                hs.append((img, img.copy()))
        hands_by_frame.append(hs)
    s = SETTINGS
    tracks = [fix_bones(smooth(fill_gaps(drop_jumps(t, s["jump"]), s["max_gap"]), s["sigma"], s["fast"]))
              for t in _tracks(hands_by_frame)]

    # face mesh with depth, lightly smoothed over the frames where it was found
    n = len(arr)
    faces: list = [None] * n
    if face_frames:
        idx = [i for i in range(min(n, len(face_frames)))
               if face_frames[i] and face_frames[i].get("landmarks_xyz")]
        if idx:
            pts = np.stack([np.asarray(face_frames[i]["landmarks_xyz"], dtype=np.float64) * px for i in idx])
            if len(idx) >= 3:
                pts = gaussian_filter1d(pts, s["face_sigma"], axis=0, mode="nearest")
            for i, p in zip(idx, pts):
                faces[i] = _r(flip(p))

    frames = []
    for i, f in enumerate(arr):
        pose = f[:25] if np.any(f[:33] != 0) else None
        frames.append({
            "h": [None if t[i] is None else _r(t[i][0]) for t in tracks],
            "w": [None if t[i] is None else _r(t[i][1]) for t in tracks],
            "p": None if pose is None else _r(flip(pose * px)),
            "f": faces[i],
        })
    return {"width": width, "height": height, "fps": fps, "n_frames": n, "frames": frames, "mirrored": mirror}
