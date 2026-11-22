#!/usr/bin/env python3
"""Real held-out validation of the two NMS thresholds in
src/mouna/coach/facial.py, which up to now were only checked by looking at
the words they're meant to detect (circular, not validation).

Method: auto_screen_references.py only extracts a face signal for the ONE
"representative" clip per word, but it screens (confidence>=0.5, correct
top-1) a much larger pool of candidate clips per word -- 72 for "what", 45
for "who", 71 for "how", 42 for "no". This script re-runs that same
screening to recover the full candidate list, then extracts a REAL face
signal for every candidate clip (not just the one representative), giving
a genuine within-type sample per word instead of n=1.

Positive set: all screened what/who/how clips (wh_question_marker) and all
screened no clips (negation_marker) -- held out from threshold-setting in
the sense that only 1 clip per word was ever looked at before.
Negative set: a random sample of non-NMS words' screened clips -- these
should NOT fire either detector; if they do, that's a real false positive.

Reports true-positive rate (recall) on the positive set and false-positive
rate on the negative set, for the CURRENT thresholds
(BROW_FURROW_THRESHOLD=0.3, HEADSHAKE_YAW_RANGE_DEG=8.0), plus the
best-separating threshold a simple sweep would pick, so the current
threshold's real quality can be judged honestly instead of assumed.
"""

import json
import random
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.insert(0, "src")
from mouna.coach.facial import (  # noqa: E402
    BROW_FURROW_NAMES,
    BROW_FURROW_THRESHOLD,
    HEADSHAKE_YAW_RANGE_DEG,
    _brow_furrow_score,
    _yaw_range,
)
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.models.transformer import PoseTransformer  # noqa: E402

MODEL_PATH = Path("data/best_model_v4canon.pth")
V4CANON_PATH = Path("data/silver/keypoints/keypoints_official_split_v4canon.parquet")
RAW_MEDIAPIPE = Path("data/silver/keypoints/keypoints_raw.parquet")
RAW_ASLC = Path("data/silver/keypoints/keypoints_asl_citizen_raw.parquet")
WLASL_VIDEOS_DIR = Path("data/raw/wlasl/videos")
ASLC_VIDEOS_DIR = Path("data/raw/asl_citizen/videos")
FACE_MODEL_PATH = Path.home() / ".mouna" / "models" / "face_landmarker.task"
SIGNS_JSON = Path("app/backend/signs.json")

MAX_SEQ_LEN = 150
CONFIDENCE_FLOOR = 0.5
GOOD_FACE_SOURCES = {"asl_citizen", "mediapipe_video"}
MAX_FRAMES = 60  # capped from 150 -- validation needs the furrow/yaw peak, not every frame
MAX_CLIPS_PER_WORD = 35  # caps runtime; still far more than the n=1 the thresholds were set on
NEG_WORDS_SAMPLE = 12
NEG_CLIPS_PER_WORD = 10
RNG_SEED = 42


def find_video_path(video_id: str, source: str) -> Path | None:
    if source == "mediapipe_video":
        p = WLASL_VIDEOS_DIR / f"{video_id}.mp4"
        return p if p.exists() else None
    if source == "asl_citizen":
        p = ASLC_VIDEOS_DIR / video_id
        return p if p.exists() else None
    return None


def extract_face_signal(video_path: Path, max_frames: int = MAX_FRAMES) -> dict | None:
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(str(video_path))
    blendshapes, yaws = [], []
    n = 0
    while n < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result.face_blendshapes:
            blendshapes.append({c.category_name: c.score for c in result.face_blendshapes[0]})
        if result.facial_transformation_matrixes:
            m = np.array(result.facial_transformation_matrixes[0])
            yaws.append(float(np.degrees(np.arctan2(-m[2, 0], np.sqrt(m[2, 1] ** 2 + m[2, 2] ** 2)))))
    cap.release()
    landmarker.close()

    if not blendshapes:
        return None
    names = list(blendshapes[0].keys())
    matrix = np.array([[bs.get(n, 0.0) for n in names] for bs in blendshapes])
    return {"blendshape_names": names, "blendshape_matrix": matrix, "head_yaw_deg": np.array(yaws, dtype=np.float32)}


def screen_candidates(word: str, v4: pd.DataFrame, raw_lookup: dict, model, idx_to_label: dict) -> list[tuple[str, str]]:
    """Returns [(video_id, source), ...] for candidates that pass the same
    confidence+correctness screen auto_screen_references.py uses -- i.e.
    the full pool, not just the 1 representative clip."""
    candidates = v4[(v4["gloss"] == word) & (v4["source"].isin(GOOD_FACE_SOURCES))]
    kept = []
    for _, row in candidates.iterrows():
        vid, source = str(row["video_id"]), row["source"]
        raw_flat = raw_lookup.get(vid)
        if raw_flat is None:
            continue
        canon = canonicalize_handedness(raw_flat, target="right")
        T = min(len(canon), MAX_SEQ_LEN)
        from mouna.data.part_normalize import normalize_per_part
        normed = normalize_per_part(canon)[:T]
        padded = normed
        if T < MAX_SEQ_LEN:
            padded = np.vstack([padded, np.zeros((MAX_SEQ_LEN - T, normed.shape[1]), dtype=np.float32)])
        x = torch.from_numpy(padded).unsqueeze(0).float()
        lengths = torch.tensor([T])
        with torch.no_grad():
            emb = model.forward_features(x, lengths)[0].numpy()
            probs = torch.softmax(model.classifier(torch.from_numpy(emb).unsqueeze(0)), dim=1)[0].numpy()
        top1 = int(np.argmax(probs))
        pred_word, conf = idx_to_label[top1], float(probs[top1])
        if pred_word == word and conf >= CONFIDENCE_FLOOR:
            kept.append((vid, source))
    return kept


def main():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = PoseTransformer(**{k: v for k, v in cfg.items() if k != "num_classes"}, num_classes=cfg["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    idx_to_label = ckpt["idx_to_label"]
    print(f"Model loaded: {cfg['num_classes']} classes", flush=True)

    v4 = pd.read_parquet(V4CANON_PATH)
    raw_mp = pd.read_parquet(RAW_MEDIAPIPE); raw_mp = raw_mp[raw_mp["success"]]
    raw_aslc = pd.read_parquet(RAW_ASLC); raw_aslc = raw_aslc[raw_aslc["success"]]
    raw_lookup = {}
    for df in (raw_mp, raw_aslc):
        for _, row in df.iterrows():
            raw_lookup[str(row["video_id"])] = pickle_loads_safe(row["keypoints_pickle"])

    with open(SIGNS_JSON) as f:
        signs = json.load(f)["signs"]
    non_nms_words = [s["word"] for s in signs if not s["nms_required"]]
    random.seed(RNG_SEED)
    neg_words = random.sample(non_nms_words, NEG_WORDS_SAMPLE)

    results = {"wh_question_marker": [], "negation_marker": [], "negative": []}

    def process(word: str, bucket: str, cap: int):
        print(f"\n--- screening '{word}' ({bucket}) ---", flush=True)
        kept = screen_candidates(word, v4, raw_lookup, model, idx_to_label)
        print(f"  {len(kept)} clips pass confidence+correctness screen", flush=True)
        random.shuffle(kept)
        kept = kept[:cap]
        for vid, source in kept:
            vp = find_video_path(vid, source)
            if vp is None:
                continue
            sig = extract_face_signal(vp)
            if sig is None:
                continue
            furrow = _brow_furrow_score(sig)
            yaw = _yaw_range(sig)
            results[bucket].append({"word": word, "video_id": vid, "furrow": furrow, "yaw_range": yaw})
        print(f"  {len([r for r in results[bucket] if r['word'] == word])} clips with real face signal extracted", flush=True)

    for w in ["what", "who", "how"]:
        process(w, "wh_question_marker", MAX_CLIPS_PER_WORD)
    process("no", "negation_marker", MAX_CLIPS_PER_WORD)
    for w in neg_words:
        process(w, "negative", NEG_CLIPS_PER_WORD)

    out_path = Path("data/nms_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results -> {out_path}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for bucket, metric_name, metric_key, thresh in [
        ("wh_question_marker", "brow furrow", "furrow", BROW_FURROW_THRESHOLD),
        ("negation_marker", "yaw range", "yaw_range", HEADSHAKE_YAW_RANGE_DEG),
    ]:
        pos = [r[metric_key] for r in results[bucket]]
        neg = [r[metric_key] for r in results["negative"]]
        if not pos or not neg:
            print(f"{bucket}: insufficient data (pos={len(pos)}, neg={len(neg)})", flush=True)
            continue
        tp_rate = float(np.mean([v >= thresh for v in pos]))
        fp_rate = float(np.mean([v >= thresh for v in neg]))
        print(f"\n{bucket} ({metric_name}, current threshold={thresh}):", flush=True)
        print(f"  positive n={len(pos)}, mean={np.mean(pos):.3f}, median={np.median(pos):.3f}", flush=True)
        print(f"  negative n={len(neg)}, mean={np.mean(neg):.3f}, median={np.median(neg):.3f}", flush=True)
        print(f"  TRUE POSITIVE RATE (recall) at current threshold: {tp_rate:.1%}", flush=True)
        print(f"  FALSE POSITIVE RATE at current threshold: {fp_rate:.1%}", flush=True)

        best_t, best_j = thresh, -1.0
        for t in np.linspace(min(pos + neg), max(pos + neg), 200):
            tpr = np.mean([v >= t for v in pos])
            fpr = np.mean([v >= t for v in neg])
            j = tpr - fpr  # Youden's J
            if j > best_j:
                best_j, best_t = j, t
        print(f"  Best-separating threshold (Youden's J): {best_t:.3f} (J={best_j:.3f} vs. current J={tp_rate - fp_rate:.3f})", flush=True)


def pickle_loads_safe(b):
    import pickle
    return pickle.loads(b)


if __name__ == "__main__":
    main()
