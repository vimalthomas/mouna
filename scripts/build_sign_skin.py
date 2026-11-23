#!/usr/bin/env python3
"""Build the 3D skin data for every practice sign (v3).

For each sign's example video: run the improved tracking from
mouna.skin.pipeline (heavy pose + video-tracked hands, gap filling, zoomed
second look, speed-aware zero-lag smoothing, metric 3D finger shape, face
mesh), and write data/references/skin/<sign>.json.gz for the Learn screen.
Raw tracker output is cached in data/references/skin_raw/ (not in git), so
re-running with new clean-up settings is fast; --redetect ignores the cache.

Needs the MediaPipe models in data/skin_models/ (not in git): hand_landmarker,
face_landmarker, pose_landmarker_heavy and pose_landmarker_lite .task files
from https://storage.googleapis.com/mediapipe-models/.

Run from repo root:  ./venv/bin/python scripts/build_sign_skin.py [sign ...] [--redetect]
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app" / "backend")]
from mouna.skin.pipeline import build, detect  # noqa: E402

OUT = ROOT / "data/references/skin"
RAW = ROOT / "data/references/skin_raw"
EXAMPLES = json.loads((ROOT / "data/references/example_videos.json").read_text())


def practice_signs() -> list[str]:
    from services.example_videos import practice_ready
    return sorted(w for w in EXAMPLES if practice_ready(w))


def compact(d: dict) -> dict:
    """Only what the app draws: per frame, hands (video px + depth), their 3D
    shape (mm), upper-body pose, face mesh."""
    frames = []
    for imp, face in zip(d["improved"], d["face"]):
        frames.append({"h": imp["hands"], "w": imp["world"], "p": imp["pose"], "f": face and face["pts"]})
    m = d["metrics"]["improved"]
    return {"video_id": d["video"].split("/")[-1][:-4], "width": d["width"], "height": d["height"], "fps": d["fps"],
            "n_frames": d["n_frames"], "frames": frames,
            "quality": {"hand_found": m["any_hand"], "face_found": d["metrics"]["face_detected"]}}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    redetect = "--redetect" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    for sign in args or practice_signs():
        raw_path = RAW / f"{sign}.pkl"
        vid = EXAMPLES[sign]["video_id"]
        raw = pickle.load(open(raw_path, "rb")) if raw_path.exists() and not redetect else None
        if raw is None or raw.get("video_id") != vid:  # example changed -> detect again
            raw = detect(sign, None)
            pickle.dump(raw, open(raw_path, "wb"))
        out = compact(build(raw))
        # stored gzipped (served as-is with Content-Encoding: gzip) to keep the repo small
        (OUT / f"{sign}.json.gz").write_bytes(gzip.compress(json.dumps(out, separators=(",", ":")).encode(), 9))
        print(f"{sign:12s} {out['width']}x{out['height']} hand {out['quality']['hand_found']:.0%} face {out['quality']['face_found']:.0%}", flush=True)


if __name__ == "__main__":
    main()
