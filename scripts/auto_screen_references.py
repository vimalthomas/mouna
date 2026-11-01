#!/usr/bin/env python3
"""Build reference entries for words with ZERO human-reviewed (confirmed-ok)
clips, using an explicitly weaker, disclosed method -- NOT by writing fake
entries into data/silver/qa/review_decisions.csv, which exists specifically
to record real human judgment and would lose its meaning if padded with
unreviewed guesses.

Method ("auto-screened", distinct from "human_reviewed"): restrict to
good-face-detection sources only (asl_citizen, mediapipe_video -- excludes
msasl_mediapipe, confirmed ~22% face-detection rate, and openpose_recovered,
which has no source video at all -- see scripts/prototype_face_landmarks.py),
then keep only clips the trained v4canon model confidently (>=0.5) and
correctly classifies. That's a real, disclosed, lower-rigor proxy for
"probably a clean, representative clip" -- not human QA, and every
resulting reference entry is tagged "qa_method": "auto_screened" so this
is visible to anything that reads the reference library later, not just
in this script's comments.

Also extracts the chosen representative clip's FACE signal (blendshapes +
head-pose) via FaceLandmarker, since the facial coach feature needs a real
per-word facial reference, not just the pose/hand one build_sign_references
already produces.

Usage: python scripts/auto_screen_references.py what who how no
"""

import pickle
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
from mouna.data.handedness import canonicalize_handedness  # noqa: E402
from mouna.data.openpose_convert import convert_video as op_convert_video  # noqa: E402
from mouna.data.part_normalize import normalize_per_part  # noqa: E402
from mouna.models.transformer import PoseTransformer  # noqa: E402

MODEL_PATH = Path("data/best_model_v4canon.pth")
V4CANON_PATH = Path("data/silver/keypoints/keypoints_official_split_v4canon.parquet")
RAW_MEDIAPIPE = Path("data/silver/keypoints/keypoints_raw.parquet")
RAW_ASLC = Path("data/silver/keypoints/keypoints_asl_citizen_raw.parquet")
REFERENCES_PATH = Path("data/references/sign_references.pkl")
WLASL_VIDEOS_DIR = Path("data/raw/wlasl/videos")
ASLC_VIDEOS_DIR = Path("data/raw/asl_citizen/videos")
FACE_MODEL_PATH = Path.home() / ".mouna" / "models" / "face_landmarker.task"

MAX_SEQ_LEN = 150
CONFIDENCE_FLOOR = 0.5
GOOD_FACE_SOURCES = {"asl_citizen", "mediapipe_video"}


def find_video_path(video_id: str, source: str) -> Path | None:
    if source == "mediapipe_video":
        p = WLASL_VIDEOS_DIR / f"{video_id}.mp4"
        return p if p.exists() else None
    if source == "asl_citizen":
        p = ASLC_VIDEOS_DIR / video_id  # video_id IS the filename for this source
        return p if p.exists() else None
    return None


def extract_face_signal(video_path: Path, max_frames: int = 150) -> dict | None:
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

    if not blendshapes:
        return None
    names = list(blendshapes[0].keys())
    matrix = np.array([[bs.get(n, 0.0) for n in names] for bs in blendshapes])
    return {
        "blendshape_names": names,
        "blendshape_matrix": matrix,  # (T, 52)
        "head_yaw_deg": np.array(yaws, dtype=np.float32),  # (T,) or shorter if matrices were missing
        "n_frames": len(blendshapes),
    }


def main():
    words = sys.argv[1:]
    if not words:
        print("Usage: python scripts/auto_screen_references.py <word> [<word> ...]")
        return

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = PoseTransformer(**{k: v for k, v in cfg.items() if k != "num_classes"}, num_classes=cfg["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    idx_to_label = ckpt["idx_to_label"]
    print(f"Model loaded: {cfg['num_classes']} classes")

    v4 = pd.read_parquet(V4CANON_PATH)
    v4["_vid"] = v4["video_id"].astype(str)

    raw_mp = pd.read_parquet(RAW_MEDIAPIPE); raw_mp = raw_mp[raw_mp["success"]]
    raw_aslc = pd.read_parquet(RAW_ASLC); raw_aslc = raw_aslc[raw_aslc["success"]]
    raw_lookup = {}
    for df in (raw_mp, raw_aslc):
        for _, row in df.iterrows():
            raw_lookup[str(row["video_id"])] = pickle.loads(row["keypoints_pickle"])

    if REFERENCES_PATH.exists():
        with open(REFERENCES_PATH, "rb") as f:
            references = pickle.load(f)
    else:
        references = {}

    for word in words:
        candidates = v4[(v4["gloss"] == word) & (v4["source"].isin(GOOD_FACE_SOURCES))]
        print(f"\n=== {word}: {len(candidates)} candidates from good-face-detection sources ===")

        embeddings, normed_seqs, raw_canon_seqs, kept_rows = [], [], [], []
        for _, row in candidates.iterrows():
            vid, source = row["_vid"], row["source"]
            raw_flat = raw_lookup.get(vid)
            if raw_flat is None:
                continue
            canon = canonicalize_handedness(raw_flat, target="right")
            normed = normalize_per_part(canon)

            T = min(len(normed), MAX_SEQ_LEN)
            padded = normed[:T]
            if T < MAX_SEQ_LEN:
                padded = np.vstack([padded, np.zeros((MAX_SEQ_LEN - T, padded.shape[1]), dtype=np.float32)])
            x = torch.from_numpy(padded).unsqueeze(0).float()
            lengths = torch.tensor([T])
            with torch.no_grad():
                emb = model.forward_features(x, lengths)[0].numpy()
                probs = torch.softmax(model.classifier(torch.from_numpy(emb).unsqueeze(0)), dim=1)[0].numpy()
            top1 = int(np.argmax(probs))
            pred_word, conf = idx_to_label[top1], float(probs[top1])

            if pred_word == word and conf >= CONFIDENCE_FLOOR:
                embeddings.append(emb)
                normed_seqs.append(normed)
                raw_canon_seqs.append(canon)
                kept_rows.append(row)

        print(f"  kept {len(embeddings)}/{len(candidates)} (correctly + confidently classified, >= {CONFIDENCE_FLOOR})")
        if not embeddings:
            print(f"  WARNING: 0 clips passed the screen for '{word}' -- leaving unreferenced, not faking it.")
            continue

        centroid = np.mean(embeddings, axis=0)
        dists = [np.linalg.norm(e - centroid) for e in embeddings]
        best_idx = int(np.argmin(dists))
        best_row = kept_rows[best_idx]

        entry = {
            "embedding_centroid": centroid,
            "representative_sequence": normed_seqs[best_idx],
            "representative_sequence_raw_canon": raw_canon_seqs[best_idx],
            "n_references": len(embeddings),
            "qa_method": "auto_screened",  # NOT human_reviewed -- see module docstring
        }

        video_path = find_video_path(best_row["_vid"], best_row["source"])
        if video_path is not None:
            face_signal = extract_face_signal(video_path)
            if face_signal is not None:
                entry["representative_face_signal"] = face_signal
                print(f"  face signal extracted: {face_signal['n_frames']} frames, "
                      f"{len(face_signal['blendshape_names'])} blendshapes")
            else:
                print(f"  WARNING: face not detected in representative clip for '{word}' -- no facial reference.")
        else:
            print(f"  WARNING: video file not found for representative clip of '{word}' -- no facial reference.")

        references[word] = entry

    with open(REFERENCES_PATH, "wb") as f:
        pickle.dump(references, f)
    print(f"\nSaved -> {REFERENCES_PATH} ({len(references)} words total)")


if __name__ == "__main__":
    main()
