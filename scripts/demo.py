#!/usr/bin/env python3
"""
Mouna Live Demo
- Press SPACE to record 3 seconds of signing
- The model predicts the sign and shows confidence
- Press Q to quit
"""

import pickle
import time
from collections import deque

import cv2
import numpy as np
import torch

from mouna.data.preprocessing import KeypointExtractor
from mouna.models.baseline import create_baseline_model

MODEL_PATH = "data/best_model.pth"
RECORD_SECONDS = 3
MAX_SEQ_LEN = 150
CONFIDENCE_THRESHOLD = 0.4  # below this → show "uncertain"


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = create_baseline_model(
        model_type="bilstm",
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"],
        dropout=cfg.get("dropout", 0.3),
        bidirectional=cfg.get("bidirectional", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    idx_to_label = checkpoint["idx_to_label"]
    return model, idx_to_label


def predict(model, idx_to_label, extractor, frames):
    """Run inference on a list of BGR frames."""
    # Extract keypoints frame by frame
    pose_seq, left_seq, right_seq = [], [], []
    for frame in frames:
        kp = extractor.extract_from_frame(frame)
        pose_seq.append(kp["pose"])
        left_seq.append(kp["left_hand"])
        right_seq.append(kp["right_hand"])

    keypoints = {
        "pose": np.array(pose_seq),
        "left_hand": np.array(left_seq),
        "right_hand": np.array(right_seq),
    }
    flat = extractor.flatten_keypoints(keypoints)
    normed = extractor.normalize_keypoints(flat)

    # Pad / truncate to MAX_SEQ_LEN
    seq_len = min(len(normed), MAX_SEQ_LEN)
    if len(normed) < MAX_SEQ_LEN:
        normed = np.vstack([normed, np.zeros((MAX_SEQ_LEN - len(normed), normed.shape[1]))])
    else:
        normed = normed[:MAX_SEQ_LEN]

    x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)       # (1, T, D)
    lengths = torch.tensor([seq_len], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = probs.max(0)

    label = idx_to_label[top_idx.item()]
    confidence = top_prob.item()

    # Top-3
    top3 = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])[:3]
    top3_labels = [(idx_to_label[i], p) for i, p in top3]

    return label, confidence, top3_labels


def draw_text(frame, text, pos, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    print("Loading model...")
    model, idx_to_label = load_model()
    print(f"Model loaded. Classes: {list(idx_to_label.values())}")

    print("Initializing MediaPipe...")
    extractor = KeypointExtractor()
    print("Ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_to_record = int(fps * RECORD_SECONDS)

    state = "idle"          # idle | countdown | recording | result
    countdown_start = None
    recorded_frames = []
    result_label = ""
    result_confidence = 0.0
    result_top3 = []
    result_time = None

    print("\nControls: SPACE = start recording | Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = display.shape[:2]

        if state == "idle":
            draw_text(display, "Press SPACE to sign", (20, 40), color=(200, 200, 200))
            if result_label:
                # Show last result
                color = (0, 220, 0) if result_confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                if result_confidence < CONFIDENCE_THRESHOLD:
                    draw_text(display, "Uncertain", (20, h - 120), scale=1.2, color=color)
                else:
                    draw_text(display, result_label.upper(), (20, h - 120), scale=1.8, color=color)
                draw_text(display, f"Confidence: {result_confidence*100:.1f}%", (20, h - 80), scale=0.7)
                for i, (lbl, p) in enumerate(result_top3):
                    draw_text(display, f"  {lbl}: {p*100:.1f}%", (20, h - 55 + i * 22), scale=0.55, color=(180, 180, 180))

        elif state == "countdown":
            elapsed = time.time() - countdown_start
            remaining = 3 - int(elapsed)
            if remaining <= 0:
                state = "recording"
                recorded_frames = []
            else:
                # Overlay countdown
                cv2.rectangle(display, (0, 0), (w, h), (0, 0, 100), 8)
                draw_text(display, f"Get ready... {remaining}", (w // 2 - 120, h // 2), scale=1.5, color=(100, 200, 255))

        elif state == "recording":
            recorded_frames.append(frame.copy())
            progress = len(recorded_frames) / frames_to_record
            bar_w = int(w * progress)
            cv2.rectangle(display, (0, h - 10), (w, h), (50, 50, 50), -1)
            cv2.rectangle(display, (0, h - 10), (bar_w, h), (0, 200, 0), -1)
            cv2.rectangle(display, (0, 0), (w, h), (0, 0, 200), 6)
            draw_text(display, "Recording...", (20, 40), scale=1.0, color=(0, 100, 255))
            secs_left = RECORD_SECONDS - progress * RECORD_SECONDS
            draw_text(display, f"{secs_left:.1f}s", (w - 100, 40), color=(0, 200, 255))

            if len(recorded_frames) >= frames_to_record:
                state = "processing"

        elif state == "processing":
            draw_text(display, "Processing...", (w // 2 - 100, h // 2), scale=1.2, color=(255, 200, 0))
            cv2.imshow("Mouna Sign Language Demo", display)
            cv2.waitKey(1)

            try:
                result_label, result_confidence, result_top3 = predict(
                    model, idx_to_label, extractor, recorded_frames
                )
            except Exception as e:
                result_label = "error"
                result_confidence = 0.0
                result_top3 = []
                print(f"Prediction error: {e}")

            state = "idle"
            result_time = time.time()
            continue

        cv2.imshow("Mouna Sign Language Demo", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            if state == "idle":
                state = "countdown"
                countdown_start = time.time()
                result_label = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
