#!/usr/bin/env python3
"""Improved tracking for the 3D skin (promoted from the fidelity lab,
lab/fidelity branch): runs today's tracking and the improved tracking on a
sign's example video and cleans the improved one up.

Two stages, so clean-up settings can be tuned without re-running trackers:
  detect  (slow, cached in lab/data/raw/<name>.pkl): every tracker's raw output
  build   (fast): today's version as-is; improved version cleaned up

today    = what the app does now: pose "lite", every frame detected from
           scratch, hands from the full frame, no clean-up.
improved = detection: pose "heavy" + hands with frame-to-frame tracking;
           frames it misses are filled from today's detection, then by a
           zoomed second look around where the hand just was.
           clean-up (whole clip is available offline, so no lag):
             1. drop one-frame jumps that are physically impossible
             2. fill short gaps from the frames on both sides
             3. zero-lag smoothing (looks both backward and forward)
             4. every finger bone keeps the clip's typical length
             5. the body's wrist is joined to the tracked hand's wrist
           also kept: each hand's true 3D finger shape (metric "world"
           landmarks) and the face mesh with depth, for the 3D skin later.
face     = 478-point mesh + 52 expression weights, lightly smoothed.

Used by scripts/build_sign_skin.py.
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "data" / "skin_models"        # MediaPipe .task files (not in git; see scripts/build_sign_skin.py)
EXAMPLES = json.loads((ROOT / "data/references/example_videos.json").read_text())
VIDEOS = ROOT / "app/wlasl/videos"

# Clean-up settings (chosen with --sweep; see tune()).
SETTINGS = {"jump": 0.8, "max_gap": 3, "sigma": 1.5, "fast": 0.3, "pose_sigma": 1.5, "face_sigma": 0.8, "image_bones": False}

# 21-point hand: parent of each joint (wrist = root)
HAND_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


# ============================ detection (slow) ============================

def _opts(cls, path, mode, **kw):
    return cls(base_options=BaseOptions(model_asset_path=str(MODELS / path)), running_mode=mode, **kw)


def _hands(result, px):
    """[(image points px (21,3), world points mm (21,3)), ...]"""
    out = []
    for h, w in zip(result.hand_landmarks, result.hand_world_landmarks):
        out.append((np.array([[l.x, l.y, l.z] for l in h]) * px,
                    np.array([[l.x, l.y, l.z] for l in w]) * 1000.0))
    return out


def _dup(h, hands, tol=0.35):
    """Same physical hand as one already found (wrists closer than 35% of
    the hand's size)?"""
    size = np.linalg.norm(h[9] - h[0]) or 1.0
    return any(np.linalg.norm(h[0] - g[0][0] if isinstance(g, tuple) else h[0] - g[0]) < tol * size for g in hands)


def _zoom_detect(tracker, rgb, near, W, H):
    """Look again for a hand around where one was a moment ago: crop that
    area, enlarge it, and detect there (helps small, low-res hands)."""
    cx, cy = near[:, 0].mean(), near[:, 1].mean()
    ext = max(np.ptp(near[:, 0]), np.ptp(near[:, 1]), 24.0)
    side = int(min(max(W, H), ext * 3.0))
    x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
    y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
    crop = rgb[y0:y0 + side, x0:x0 + side]
    if crop.size == 0:
        return None
    zoom = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_CUBIC)
    r = tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(zoom)))
    if not r.hand_landmarks:
        return None
    cw, ch = crop.shape[1], crop.shape[0]
    img = np.array([[x0 + l.x * cw, y0 + l.y * ch, l.z * cw] for l in r.hand_landmarks[0]])
    world = np.array([[l.x, l.y, l.z] for l in r.hand_world_landmarks[0]]) * 1000.0
    return img, world


def detect(sign: str, video_id: str | None) -> dict:
    c = dict(EXAMPLES[sign])
    if video_id:
        c["video_id"] = video_id
    cap = cv2.VideoCapture(str(VIDEOS / f"{c['video_id']}.mp4"))
    W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    px = np.array([W, H, W], dtype=np.float64)  # MediaPipe z uses x's scale
    n_expected = 2 if c.get("n_hands") == 2 else 1
    image = lambda rgb: mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))  # noqa: E731

    def hand_opts(mode, n=2, conf=0.5):
        return _opts(vision.HandLandmarkerOptions, "hand_landmarker.task", mode, num_hands=n,
                     min_hand_detection_confidence=conf, min_hand_presence_confidence=conf)

    # today's pipeline
    pose_t = vision.PoseLandmarker.create_from_options(_opts(
        vision.PoseLandmarkerOptions, "pose_landmarker_lite.task", vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5))
    hand_t = vision.HandLandmarker.create_from_options(hand_opts(vision.RunningMode.IMAGE))
    today_hands, today_poses = [], []
    for rgb in frames:
        pr, hr = pose_t.detect(image(rgb)), hand_t.detect(image(rgb))
        today_poses.append(np.array([[l.x, l.y, l.z] for l in pr.pose_landmarks[0]]) * px if pr.pose_landmarks else None)
        today_hands.append(_hands(hr, px))

    # improved detection
    pose_t = vision.PoseLandmarker.create_from_options(_opts(
        vision.PoseLandmarkerOptions, "pose_landmarker_heavy.task", vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5))
    hand_t = vision.HandLandmarker.create_from_options(hand_opts(vision.RunningMode.VIDEO))
    zoom_t = vision.HandLandmarker.create_from_options(hand_opts(vision.RunningMode.IMAGE, 1, 0.4))
    found, poses, source = [], [], {"video": 0, "today": 0, "zoom": 0}
    for i, rgb in enumerate(frames):
        ts = int(i * 1000 / fps)
        pr, hr = pose_t.detect_for_video(image(rgb), ts), hand_t.detect_for_video(image(rgb), ts)
        poses.append(np.array([[l.x, l.y, l.z] for l in pr.pose_landmarks[0]]) * px if pr.pose_landmarks else None)
        hands = _hands(hr, px)
        source["video"] += len(hands)
        for h in today_hands[i]:
            if len(hands) >= n_expected or _dup(h[0], hands):
                continue
            hands.append(h); source["today"] += 1
        found.append(hands)
    for i in range(len(frames)):
        if len(found[i]) >= n_expected:
            continue
        for j in sorted(range(max(0, i - 6), min(len(frames), i + 7)), key=lambda j: abs(j - i)):
            cand = [h for h in found[j] if not _dup(h[0], found[i])]
            if j == i or not cand:
                continue
            h = _zoom_detect(zoom_t, frames[i], cand[0][0], W, H)
            if h is not None and not _dup(h[0], found[i]):
                found[i].append(h); source["zoom"] += 1
            break

    face_t = vision.FaceLandmarker.create_from_options(_opts(
        vision.FaceLandmarkerOptions, "face_landmarker.task", vision.RunningMode.VIDEO,
        output_face_blendshapes=True, num_faces=1))
    faces = []
    for i, rgb in enumerate(frames):
        r = face_t.detect_for_video(image(rgb), int(i * 1000 / fps))
        faces.append(None if not r.face_landmarks else (
            np.array([[l.x, l.y, l.z] for l in r.face_landmarks[0]]) * px,
            {b.category_name: b.score for b in r.face_blendshapes[0]}))
    return {"sign": sign, "video_id": c["video_id"], "W": W, "H": H, "fps": fps, "n": len(frames),
            "today_hands": today_hands, "today_poses": today_poses, "found": found, "poses": poses,
            "faces": faces, "source": source}


# ============================== clean-up (fast) ===========================

def _match_slots(prev, hands):
    """Keep each physical hand in the same slot (0/1) by wrist distance."""
    slots = [None, None]
    if not hands:
        return slots
    w = lambda h: h[0][0]  # noqa: E731 -- wrist of the image points
    if len(hands) == 1:
        h = hands[0]
        if prev[1] is not None and (prev[0] is None or np.linalg.norm(w(h) - w(prev[1])) < np.linalg.norm(w(h) - w(prev[0]))):
            slots[1] = h
        else:
            slots[0] = h
        return slots
    a, b = hands[:2]
    if prev[0] is not None and prev[1] is not None:
        keep = np.linalg.norm(w(a) - w(prev[0])) + np.linalg.norm(w(b) - w(prev[1]))
        swap = np.linalg.norm(w(a) - w(prev[1])) + np.linalg.norm(w(b) - w(prev[0]))
        return [b, a] if swap < keep else [a, b]
    return sorted([a, b], key=lambda h: w(h)[0])


def _tracks(hands_by_frame):
    """Per-slot lists of (image, world) or None per frame."""
    prev, slots = [None, None], []
    for hands in hands_by_frame:
        s = _match_slots(prev, hands)
        prev = [x if x is not None else p for x, p in zip(s, prev)]
        slots.append(s)
    return [[s[k] for s in slots] for k in range(2)]


def _size(track):
    v = [np.linalg.norm(h[0][9] - h[0][0]) for h in track if h is not None]
    return float(np.median(v)) if v else 1.0


def drop_jumps(track, jump):
    """A detection that leaps away and straight back (wrist moves more than
    `jump` hand-sizes from both neighbours, which agree with each other) is
    a tracking error, not motion."""
    size, out = _size(track), list(track)
    for i in range(1, len(track) - 1):
        a, b, c = track[i - 1], track[i], track[i + 1]
        if a is None or b is None or c is None:
            continue
        wa, wb, wc = a[0][0], b[0][0], c[0][0]
        if (np.linalg.norm(wb - wa) > jump * size and np.linalg.norm(wb - wc) > jump * size
                and np.linalg.norm(wa - wc) < jump * size):
            out[i] = None
    return out


def fill_gaps(track, max_gap):
    """Short dropouts (<= max_gap frames) are filled by blending the frames
    on either side, so a hand doesn't blink out mid-sign."""
    out, idx = list(track), [i for i, h in enumerate(track) if h is not None]
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                out[i] = tuple((1 - t) * x + t * y for x, y in zip(track[a], track[b]))
    return out


def smooth(track, sigma, fast=0.0):
    """Zero-lag smoothing over each continuous run: every frame is averaged
    with frames before AND after it, so the hand doesn't trail its motion.
    With `fast` > 0 it is also speed-aware: while the hand moves faster than
    `fast` hand-sizes per frame it is smoothed only lightly, so quick moves
    keep their shape; when still, it gets the full smoothing."""
    if sigma <= 0:
        return list(track)
    if fast > 0:
        strong, light = smooth(track, sigma), smooth(track, min(0.5, sigma))
        size, out = _size(track), list(track)
        for i, h in enumerate(track):
            if h is None:
                continue
            nb = [track[j][0][0] for j in (i - 1, i + 1) if 0 <= j < len(track) and track[j] is not None]
            speed = max((np.linalg.norm(h[0][0] - w) for w in nb), default=0.0) / size
            a = min(1.0, speed / fast)  # 0 = still -> strong, 1 = fast -> light
            out[i] = tuple((1 - a) * x + a * y for x, y in zip(strong[i], light[i]))
        return out
    out, i = list(track), 0
    while i < len(track):
        if track[i] is None:
            i += 1
            continue
        j = i
        while j < len(track) and track[j] is not None:
            j += 1
        if j - i >= 3:
            for part in range(2):  # image points, world points
                arr = np.stack([track[k][part] for k in range(i, j)])
                sm = gaussian_filter1d(arr, sigma, axis=0, mode="nearest")
                for k in range(i, j):
                    out[k] = (sm[k - i], out[k][1]) if part == 0 else (out[k][0], sm[k - i])
        i = j
    return out


def fix_bones(track, parts=(1,)):
    """Every bone keeps its direction per frame but gets the clip's typical
    length, so fingers never stretch or shrink. Applied to the true 3D shape
    (world, part 1) by default -- NOT to the flat image points: on video a
    finger pointing at the camera really does look shorter, and forcing its
    length there bent the drawing away from the real hand (~10% drift)."""
    out = list(track)
    for part in parts:
        pts = [h[part] for h in out if h is not None]
        if len(pts) < 3:
            return out
        A = np.stack(pts)
        L = [np.median(np.linalg.norm(A[:, j] - A[:, HAND_PARENT[j]], axis=1)) if j else 0.0 for j in range(21)]
        for i, h in enumerate(out):
            if h is None:
                continue
            f, g = h[part], h[part].copy()
            for j in range(1, 21):
                d = f[j] - f[HAND_PARENT[j]]
                n = np.linalg.norm(d)
                g[j] = g[HAND_PARENT[j]] + (d / n * L[j] if n > 1e-6 else d)
            out[i] = (g, h[1]) if part == 0 else (h[0], g)
    return out


def clean_poses(poses, sigma, max_gap):
    tr = fill_gaps([None if p is None else (p,) for p in poses], max_gap)
    out, i = [None if t is None else t[0] for t in tr], 0
    while i < len(out):
        if out[i] is None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is not None:
            j += 1
        if j - i >= 3 and sigma > 0:
            sm = gaussian_filter1d(np.stack(out[i:j]), sigma, axis=0, mode="nearest")
            out[i:j] = list(sm)
        i = j
    return out


def join_wrists(poses, tracks):
    """Move the body's wrist (pose 15/16) onto the tracked hand's wrist when
    they are the same hand, so drawn hands sit on their arms."""
    out = []
    for i, p in enumerate(poses):
        if p is None:
            out.append(None)
            continue
        p = p.copy()
        sh = np.linalg.norm(p[11, :2] - p[12, :2]) or 1.0
        for tr in tracks:
            h = tr[i]
            if h is None:
                continue
            w = h[0][0]
            k = min((15, 16), key=lambda k: np.linalg.norm(p[k, :2] - w[:2]))
            if np.linalg.norm(p[k, :2] - w[:2]) < 0.6 * sh:
                p[k, :2] = w[:2]
        out.append(p)
    return out


# ================================ metrics =================================

def jitter(seq, scale, still=0.15):
    """Shaking: frame-to-frame wobble on frames where the hand is nearly
    still (moving < `still` hand-sizes per frame). Real fast motion is
    excluded, so this measures noise, not signing."""
    acc = []
    for a, b, c in zip(seq, seq[1:], seq[2:]):
        if a is None or b is None or c is None:
            continue
        if scale and np.linalg.norm(c[0] - a[0]) / 2 / scale > still:
            continue
        acc.append(np.linalg.norm(a - 2 * b + c, axis=-1).mean())
    return float(np.mean(acc) / scale) if acc and scale else None


def bone_cv(seq):
    pts = [f for f in seq if f is not None]
    if len(pts) < 3:
        return None
    A = np.stack(pts)
    cvs = [np.linalg.norm(A[:, j] - A[:, HAND_PARENT[j]], axis=1) for j in range(1, 21)]
    return float(np.mean([L.std() / L.mean() for L in cvs if L.mean() > 1e-6]))


def drift(clean, raw, size):
    """How far the cleaned hand sits from what the tracker actually saw, on
    frames where it saw one (share of hand size; lower = more faithful)."""
    d = [np.linalg.norm(c[0] - r[0], axis=1).mean() for c, r in zip(clean, raw) if c is not None and r is not None]
    return float(np.mean(d) / size) if d else None


def hand_metrics(tracks, raw_tracks=None):
    n = len(tracks[0])
    m = {"any_hand": round(sum(any(t[i] is not None for t in tracks) for i in range(n)) / max(n, 1), 3),
         "both_hands": round(sum(all(t[i] is not None for t in tracks) for i in range(n)) / max(n, 1), 3)}
    for k, tr in enumerate(tracks):
        img = [None if h is None else h[0] for h in tr]
        if sum(x is not None for x in img) < 3:
            continue
        size = _size(tr)
        m[f"hand{k}"] = {"detected": round(sum(x is not None for x in img) / n, 3),
                         "jitter": round(jitter(img, size) or 0, 4), "bone_cv": round(bone_cv(img) or 0, 4)}
        if raw_tracks is not None:
            m[f"hand{k}"]["drift"] = round(drift(tr, raw_tracks[k], size) or 0, 4)
    return m


def arm_metrics(poses):
    seq = [None if p is None else p[11:17, :2] for p in poses]
    ok = [p for p in seq if p is not None]
    sh = float(np.median([np.linalg.norm(p[0] - p[1]) for p in ok])) if ok else 0.0
    return {"detected": round(len(ok) / max(len(seq), 1), 3), "jitter": round(jitter(seq, sh) or 0, 4)}


# ================================= build ==================================

def build(raw: dict, s: dict = SETTINGS) -> dict:
    W, H, n = raw["W"], raw["H"], raw["n"]
    out = {"sign": raw["sign"], "video": f"videos/{raw['video_id']}.mp4", "width": W, "height": H,
           "fps": raw["fps"], "n_frames": n, "settings": s, "current": [], "improved": [], "face": []}

    today = _tracks(raw["today_hands"])
    raw_tracks = _tracks(raw["found"])
    tracks = [fix_bones(smooth(fill_gaps(drop_jumps(t, s["jump"]), s["max_gap"]), s["sigma"], s.get("fast", 0.0)),
                        (0, 1) if s.get("image_bones") else (1,)) for t in raw_tracks]
    poses = join_wrists(clean_poses(raw["poses"], s["pose_sigma"], s["max_gap"]), tracks)

    def frame(pose, hands, world=False):
        f = {"pose": None if pose is None else np.round(pose[:25, :3], 1).tolist(),  # upper body, with depth
             "hands": [None if h is None else np.round(h[0], 1).tolist() for h in hands]}
        if world:
            f["world"] = [None if h is None else np.round(h[1], 1).tolist() for h in hands]
        return f
    out["current"] = [frame(raw["today_poses"][i], [today[0][i], today[1][i]]) for i in range(n)]
    out["improved"] = [frame(poses[i], [tracks[0][i], tracks[1][i]], world=True) for i in range(n)]

    # face: light zero-lag smoothing of the mesh and expression weights
    idx = [i for i, f in enumerate(raw["faces"]) if f is not None]
    if idx:
        pts = np.stack([raw["faces"][i][0] for i in idx])
        names = list(raw["faces"][idx[0]][1])
        bs = np.array([[raw["faces"][i][1][k] for k in names] for i in idx])
        if s["face_sigma"] > 0 and len(idx) >= 3:
            pts = gaussian_filter1d(pts, s["face_sigma"], axis=0, mode="nearest")
            bs = gaussian_filter1d(bs, s["face_sigma"], axis=0, mode="nearest")
        face = {i: (p, b) for i, p, b in zip(idx, pts, bs)}
        out["face"] = [None if i not in face else {"pts": np.round(face[i][0], 1).tolist(),
                                                   "bs": {k: round(float(v), 3) for k, v in zip(names, face[i][1])}}
                       for i in range(n)]
    else:
        out["face"] = [None] * n

    out["metrics"] = {
        "current": {**hand_metrics(today), "arms": arm_metrics(raw["today_poses"])},
        "improved": {**hand_metrics(tracks, raw_tracks), "arms": arm_metrics(poses)},
        "hand_sources": raw["source"],
        "face_detected": round(len(idx) / max(n, 1), 3),
    }
    return out
