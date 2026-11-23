#!/usr/bin/env python3
"""Pick, per sign, the example video the learner watches so that it AGREES
with the reference the learner is graded against.

Why: WLASL often has several variants of a sign (e.g. 26 "go" videos), and
the app used to pick the shortest video regardless of variant. Grading each
sign's shown example against its own reference found 6 of 23 examples graded
as a *different* sign (go, accident, africa, can, need, what) -- a learner
copying the video could not pass. Here every on-disk candidate is run
through the same live pipeline (keypoint_service at the app's 10 fps capture
rate -> model_service.score_against_target) and the best-agreeing one wins.

Also records, from that chosen video itself, where the hand is (bbox, for
the zoomed prep stills) and when the hands are up (for picking still
frames) -- previously the crop came from a different recording.

Output: data/references/example_videos.json, read by signs_service.get_video
and the /signs-with-reference practice-set endpoint. Run from repo root:
    ./venv/bin/python scripts/pick_example_videos.py [max_candidates_per_sign]
    ./venv/bin/python scripts/pick_example_videos.py --paths-only   # refresh paths for current picks
"""
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
sys.path[:0] = [str(BACKEND), str(ROOT / "src")]

import os  # noqa: E402

os.chdir(BACKEND)  # config paths (videos_dir, .env) are relative to the backend dir

from config import settings  # noqa: E402
from services.keypoint_service import keypoint_service  # noqa: E402
from services.model_service import model_service  # noqa: E402
from services.signs_service import signs_service  # noqa: E402

OUTPUT = ROOT / "data" / "references" / "example_videos.json"
CAPTURE_FPS = 10  # matches camera-view.component.ts


def keypoints_for(entry: dict):
    path = Path(settings.videos_dir) / f"{entry['video_id']}.mp4"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or entry.get("fps") or 25
    step = max(1, round(fps / CAPTURE_FPS))
    start = max(0, (entry.get("frame_start") or 1) - 1)
    end = entry.get("frame_end")
    end = None if end in (None, -1) else end
    kps, times, i = [], [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0 and i >= start and (end is None or i <= end):
            jpg = cv2.imencode(".jpg", cv2.resize(img, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
            kps.append(keypoint_service.extract_from_base64(base64.b64encode(jpg).decode()))
            times.append(i / fps)
        i += 1
    return kps, times


def hand_info(kps: list, times: list):
    """bbox of all detected hand points (normalized to the full video frame,
    which is what the 640x480 resize preserves) + seconds the hands are up."""
    arr = np.asarray(kps, dtype=np.float32).reshape(len(kps), 75, 3)
    hands = arr[:, 33:75, :2]
    present = np.any(hands != 0, axis=(1, 2))
    if not present.any():
        return None, None
    pts = hands[present].reshape(-1, 2)
    pts = pts[np.any(pts != 0, axis=1)]
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.04
    bbox = {
        "x_min": float(max(0, lo[0] - pad)), "x_max": float(min(1, hi[0] + pad)),
        "y_min": float(max(0, lo[1] - pad)), "y_max": float(min(1, hi[1] + pad)),
    }
    idx = np.where(present)[0]
    return bbox, {"start": float(times[idx[0]]), "end": float(times[idx[-1]])}


PALM_IDX = [0, 5, 9, 13, 17]  # wrist + knuckles: a stable "where the hand is" point


def _hand_tracks(arr: np.ndarray) -> list[list[tuple[int, np.ndarray]]]:
    """Follow each physical hand by POSITION, not by MediaPipe's left/right
    label: the label can flip between frames (seen on accident/all/now,
    where both drawn paths ran the same way). Per frame, every detected hand
    (either label slot) is matched to the nearest existing track; a new
    track starts only when a hand is far from both. Returns up to two tracks
    of (frame index, palm point)."""
    tracks: list[list[tuple[int, np.ndarray]]] = []
    for f in range(arr.shape[0]):
        found = []
        for sl in (slice(33, 54), slice(54, 75)):
            h = arr[f, sl, :2]
            if np.any(h != 0):
                found.append(h[PALM_IDX].mean(axis=0))
        if not found:
            continue
        if len(found) == 2 and np.linalg.norm(found[0] - found[1]) < 0.03:
            found = [found[0]]  # same hand detected twice
        last = [t[-1][1] for t in tracks]
        if len(found) == 2 and len(tracks) == 2:
            # pick the pairing with the smaller total jump
            keep = np.linalg.norm(found[0] - last[0]) + np.linalg.norm(found[1] - last[1])
            swap = np.linalg.norm(found[0] - last[1]) + np.linalg.norm(found[1] - last[0])
            pair = [(0, 0), (1, 1)] if keep <= swap else [(0, 1), (1, 0)]
            for fi, ti in pair:
                tracks[ti].append((f, found[fi]))
            continue
        for pt in found:
            if not tracks:
                tracks.append([(f, pt)])
                continue
            d = [np.linalg.norm(pt - t[-1][1]) for t in tracks]
            i = int(np.argmin(d))
            # A lone detection far from the only track starts a second hand
            # only for a very large jump: fast one-handed signs (MAN: forehead
            # -> chest in ~1s) moved >0.15 between frames and got split into
            # two short tracks, too short to draw.
            if len(tracks) < 2 and d[i] > (0.15 if len(found) == 2 else 0.35):
                tracks.append([(f, pt)])
            else:
                tracks[i].append((f, pt))
    return tracks


def hand_paths(kps: list, times: list, n_points: int = 40) -> list[dict]:
    """Path of each hand through the sign, in the video's own normalized
    [0,1] image coords (same frame the stills/video are drawn in), for the
    direction arrows in the app. Hands are followed by position (see
    _hand_tracks); a track is kept if it covers at least 20% of frames
    (min 6; hands at the face are often occluded). Rest stretches at either
    end are trimmed, then the path is lightly smoothed and resampled.
    `still` marks signs made in one spot (tiny travel): the app draws a
    "stays here" ring instead of an arrow for those."""
    arr = np.asarray(kps, dtype=np.float32).reshape(len(kps), 75, 3)
    t = np.asarray(times, dtype=np.float32)
    out = []
    for track in _hand_tracks(arr):
        if len(track) < max(6, 0.2 * len(arr)):
            continue
        idx = np.array([f for f, _ in track])
        pts = np.stack([p for _, p in track])
        tt = t[idx]
        # Drop the raise-from-rest and lower-to-rest stretches at either end:
        # leading/trailing frames whose hand sits in the lowest quarter of its
        # own vertical range (image y grows downward). Otherwise the arrows
        # end pointing down at the signer's lap instead of along the sign.
        y = pts[:, 1]
        rest_line = y.max() - 0.25 * (y.max() - y.min())
        up = np.where(y < rest_line)[0]
        if len(up) >= 4:
            pts, tt = pts[up[0]:up[-1] + 1], tt[up[0]:up[-1] + 1]
        if len(pts) >= 3:  # light smoothing, keeps the ends in place
            sm = pts.copy()
            sm[1:-1] = (pts[:-2] + pts[1:-1] + pts[2:]) / 3
            pts = sm
        grid = np.linspace(tt[0], tt[-1], n_points)
        xs = np.interp(grid, tt, pts[:, 0])
        ys = np.interp(grid, tt, pts[:, 1])
        extent = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
        out.append({
            "t": [round(float(v), 3) for v in grid],
            "x": [round(float(v), 4) for v in xs],
            "y": [round(float(v), 4) for v in ys],
            "still": extent < 0.05,
        })
    out.sort(key=lambda p: np.mean(p["x"]))  # screen left first
    for p, side in zip(out, ("screen_left", "screen_right") if len(out) == 2 else ("only",)):
        p["hand"] = side
    return out


def count_hands(kps: list) -> int:
    """1 or 2: how many hands the signer uses in this video. Counts frames
    where BOTH hand slots are detected, before any path merging (signs like
    bed press the hands together, which the path tracker merges as one).
    Two hands if both are seen in at least 30% of the frames with a hand."""
    arr = np.asarray(kps, dtype=np.float32).reshape(len(kps), 75, 3)
    l = np.any(arr[:, 33:54] != 0, axis=(1, 2))
    r = np.any(arr[:, 54:75] != 0, axis=(1, 2))
    any_hand = (l | r).sum()
    return 2 if any_hand and (l & r).sum() / any_hand >= 0.3 else 1


def both_hands_share(kps: list) -> float:
    """Share of frames with a hand where BOTH hands are detected."""
    arr = np.asarray(kps, dtype=np.float32).reshape(len(kps), 75, 3)
    l = np.any(arr[:, 33:54] != 0, axis=(1, 2))
    r = np.any(arr[:, 54:75] != 0, axis=(1, 2))
    any_hand = (l | r).sum()
    return round(float((l & r).sum() / any_hand), 3) if any_hand else 0.0


def rank(score: dict) -> tuple:
    params = score.get("parameters") or {}
    # A part signers split on ("uncertain", match None) can't contradict the
    # example, so it counts as agreeing; only a real mismatch counts against.
    matched = sum(1 for p in params.values() if p.get("match") is not False)
    return (
        score.get("overall_verdict") == "good",
        bool(score.get("hand_count_match")),
        matched,
        score.get("embedding_similarity") or -1.0,
    )


def paths_only() -> None:
    """Recompute hand paths (+ bbox / hands-up window) for the ALREADY chosen
    example of each sign, without re-grading every candidate."""
    keypoint_service.initialize()
    signs_service.load()
    data = json.loads(OUTPUT.read_text())
    for word, c in data.items():
        entry = next((e for e in signs_service.video_index.get(word, []) if e["video_id"] == c["video_id"]), None)
        if entry is None:
            print(f"{word:12s} chosen video not indexed, skipped")
            continue
        kps, times = keypoints_for(entry)
        c["hand_bbox"], c["hands_up_seconds"] = hand_info(kps, times)
        c["hand_paths"] = hand_paths(kps, times)
        c["n_hands"] = count_hands(kps)
        c["both_hands_share"] = both_hands_share(kps)
        print(f"{word:12s} {len(c['hand_paths'])} hand path(s), {c['n_hands']} hand(s)", flush=True)
    OUTPUT.write_text(json.dumps(data, indent=2))
    print(f"Saved -> {OUTPUT}")


def main():
    if "--paths-only" in sys.argv:
        paths_only()
        return
    max_candidates = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    model_service.load_model()
    keypoint_service.initialize()
    signs_service.load()

    words = sorted(w for w in model_service.references if model_service.get_reference_buckets(w) is not None)
    out = {}
    for word in words:
        entries = signs_service.video_index.get(word, [])[:max_candidates]
        best = None
        for entry in entries:
            kps, times = keypoints_for(entry)
            if len(kps) < 5:
                continue
            score = model_service.score_against_target(kps, word)
            if score.get("embedding_similarity") is None:
                continue
            r = rank(score)
            if best is None or r > best[0]:
                best = (r, entry, score, kps, times)
        if best is None:
            print(f"{word:12s} no usable candidate")
            continue
        r, entry, score, kps, times = best
        bbox, hands_up = hand_info(kps, times)
        out[word] = {
            "video_id": entry["video_id"],
            "frame_start": entry.get("frame_start"),
            "frame_end": entry.get("frame_end"),
            "fps": entry.get("fps"),
            "verdict": score["overall_verdict"],
            "similarity": round(float(score["embedding_similarity"]), 3),
            "hand_count_match": score["hand_count_match"],
            "params_matched": r[2],
            "hand_bbox": bbox,
            "hands_up_seconds": hands_up,
            "hand_paths": hand_paths(kps, times),
            "n_hands": count_hands(kps),
            "both_hands_share": both_hands_share(kps),
            "candidates_checked": len(entries),
        }
        print(f"{word:12s} {entry['video_id']:>6s} verdict={score['overall_verdict']:10s} sim={score['embedding_similarity']:.2f} "
              f"hands={score['hand_count_match']} params={r[2]}/3 (of {len(entries)} candidates)", flush=True)

    OUTPUT.write_text(json.dumps(out, indent=2))
    good = [w for w, v in out.items() if v["verdict"] == "good" and v["hand_count_match"]]
    print(f"\nPractice-ready (example passes its own grading): {len(good)}/{len(out)}: {good}")
    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
