"""STANDALONE FEASIBILITY PROTOTYPE -- not part of the production pipeline.

Investigates whether MediaPipe's FaceLandmarker can plug into this project's
existing keypoint-extraction pattern (see src/mouna/data/preprocessing.py's
KeypointExtractor / _get_model_path) to finally capture non-manual markers
(facial expression, brow raise/furrow, mouth morphemes, head tilt) -- the
one ASL phonological parameter (of handshape/location/movement/orientation/
non-manual) this pipeline has never extracted.

Does NOT touch preprocessing.py, any dataset-build script, or anything in
the training pipeline. Run directly:

    python scripts/prototype_face_landmarks.py

Reports, per video and in aggregate:
  - detection rate (fraction of frames with a face found) -- the same
    signal handedness.py's _hand_activity implicitly measures for hands
  - landmark count returned (expect 478 for FaceLandmarker)
  - whether output_face_blendshapes / output_facial_transformation_matrixes
    work, and what the blendshape vector looks like
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Same model-cache pattern as preprocessing.py's _MODEL_URLS / _get_model_path.
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_CACHE_DIR = Path.home() / ".mouna" / "models"

FACE_LANDMARKS_EXPECTED = 478

# A handful of real sample videos across sources. Picked by just listing
# each source dir and taking the first N -- no cherry-picking for clean faces.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WLASL_DIR = PROJECT_ROOT / "data/raw/wlasl/videos"
ASLC_DIR = PROJECT_ROOT / "data/raw/asl_citizen/videos"
MSASL_DIR = PROJECT_ROOT / "data/raw/msasl_official/videos"

N_PER_SOURCE = 5


def _get_face_model_path() -> str:
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _MODEL_CACHE_DIR / "face_landmarker.task"
    if not local_path.exists():
        print(f"Downloading face_landmarker.task to {local_path} ...")
        urllib.request.urlretrieve(_FACE_MODEL_URL, local_path)
        print("Downloaded.")
    else:
        print(f"Using cached model at {local_path}")
    return str(local_path)


def build_face_landmarker(model_path: str, with_blendshapes: bool) -> vision.FaceLandmarker:
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=with_blendshapes,
        output_facial_transformation_matrixes=with_blendshapes,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def pick_sample_videos():
    samples = []
    for d, label in [(WLASL_DIR, "wlasl"), (ASLC_DIR, "asl_citizen"), (MSASL_DIR, "msasl_official")]:
        if not d.exists():
            print(f"WARNING: {d} does not exist, skipping source '{label}'")
            continue
        vids = sorted(d.glob("*.mp4"))[:N_PER_SOURCE]
        for v in vids:
            samples.append((label, v))
    return samples


def analyze_video(landmarker: vision.FaceLandmarker, video_path: Path, max_frames: int = 90):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = 0
    detected_frames = 0
    landmark_counts = []
    blendshape_names_seen = None
    blendshape_sample = None
    had_transform_matrix = False

    while True:
        if total_frames >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            detected_frames += 1
            landmark_counts.append(len(result.face_landmarks[0]))

            if result.face_blendshapes:
                bs = result.face_blendshapes[0]
                if blendshape_names_seen is None:
                    blendshape_names_seen = [c.category_name for c in bs]
                if blendshape_sample is None:
                    blendshape_sample = {c.category_name: round(c.score, 4) for c in bs}

            if result.facial_transformation_matrixes:
                had_transform_matrix = True

    cap.release()

    return {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detect_rate": detected_frames / total_frames if total_frames else 0.0,
        "landmark_counts": set(landmark_counts),
        "blendshape_names": blendshape_names_seen,
        "blendshape_sample": blendshape_sample,
        "had_transform_matrix": had_transform_matrix,
    }


def main():
    model_path = _get_face_model_path()

    print("\n=== Building FaceLandmarker with blendshapes + transform matrixes enabled ===")
    try:
        landmarker = build_face_landmarker(model_path, with_blendshapes=True)
        blendshapes_supported = True
    except Exception as e:
        print(f"Blendshape-enabled landmarker failed to build: {e}")
        print("Falling back to landmarks-only landmarker.")
        landmarker = build_face_landmarker(model_path, with_blendshapes=False)
        blendshapes_supported = False

    samples = pick_sample_videos()
    print(f"\nRunning on {len(samples)} videos across sources: "
          f"{sorted(set(s[0] for s in samples))}\n")

    results = []
    t0 = time.time()
    for label, path in samples:
        r = analyze_video(landmarker, path)
        r["source"] = label
        r["video"] = path.name
        results.append(r)
        print(
            f"[{label:14s}] {path.name:40.40s} "
            f"frames={r['total_frames']:3d} detect_rate={r['detect_rate']:.2%} "
            f"landmark_counts={r['landmark_counts']} "
            f"blendshapes={'yes' if r['blendshape_names'] else 'no'} "
            f"transform_matrix={'yes' if r['had_transform_matrix'] else 'no'}"
        )
    elapsed = time.time() - t0

    print("\n=== Aggregate ===")
    rates = [r["detect_rate"] for r in results if r["total_frames"] > 0]
    print(f"Videos processed: {len(results)}")
    print(f"Mean detect_rate: {np.mean(rates):.2%}" if rates else "No frames processed.")
    print(f"Min/Max detect_rate: {min(rates):.2%} / {max(rates):.2%}" if rates else "")
    by_source = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r["detect_rate"])
    for src, vals in by_source.items():
        print(f"  {src}: mean detect_rate={np.mean(vals):.2%} over {len(vals)} videos")

    all_counts = set()
    for r in results:
        all_counts |= r["landmark_counts"]
    print(f"Distinct landmark counts observed across all videos/frames: {all_counts} "
          f"(expected {FACE_LANDMARKS_EXPECTED})")

    print(f"\nBlendshapes supported by this mediapipe build: {blendshapes_supported}")
    sample_with_bs = next((r for r in results if r["blendshape_names"]), None)
    if sample_with_bs:
        print(f"Blendshape vector length: {len(sample_with_bs['blendshape_names'])}")
        print("First 15 blendshape names:", sample_with_bs["blendshape_names"][:15])
        print("Sample scores (one frame):")
        for k, v in list(sample_with_bs["blendshape_sample"].items())[:15]:
            print(f"    {k}: {v}")
    else:
        print("No video produced a blendshape result (either unsupported, or face never detected "
              "in the frames that had blendshapes enabled).")

    print(f"\nTotal wall time for {len(results)} videos "
          f"({sum(r['total_frames'] for r in results)} frames): {elapsed:.1f}s")


if __name__ == "__main__":
    main()
