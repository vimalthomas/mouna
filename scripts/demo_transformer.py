#!/usr/bin/env python3
"""
Mouna Live Demo — PoseTransformer
=====================================
Controls:
  SPACE  — start a 3-second recording, then predict
  Q/ESC  — quit

Features:
  - Live skeleton overlay (pose + both hands) while recording
  - Top-3 prediction with confidence bars
  - Works with best_model_transformer.pth
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time

import cv2
import numpy as np
import torch

from mouna.data.preprocessing import KeypointExtractor
from mouna.models.transformer import PoseTransformer

MODEL_PATH    = "data/best_model_transformer.pth"
RECORD_SECS   = 2
MAX_SEQ_LEN   = 150
CONF_THRESH   = 0.30   # below → "uncertain"

# ── Skeleton connectivity ────────────────────────────────────────────────────
_POSE_LINKS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
]
_HAND_LINKS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = PoseTransformer(
        input_dim      = cfg["input_dim"],
        d_model        = cfg["d_model"],
        nhead          = cfg["nhead"],
        num_layers     = cfg["num_layers"],
        dim_feedforward= cfg["dim_feedforward"],
        dropout        = cfg.get("dropout", 0.3),
        num_classes    = cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["idx_to_label"]


# ── Skeleton drawing ──────────────────────────────────────────────────────────
def draw_skeleton(frame, kp):
    """kp = dict with 'pose'(33,3), 'left_hand'(21,3), 'right_hand'(21,3) in [0,1]."""
    h, w = frame.shape[:2]

    def px(pt):
        return (int(pt[0] * w), int(pt[1] * h))

    pose = kp["pose"]
    for i, j in _POSE_LINKS:
        if pose[i].any() and pose[j].any():
            cv2.line(frame, px(pose[i]), px(pose[j]), (0, 200, 100), 2)
    for i in range(len(pose)):
        if pose[i].any():
            cv2.circle(frame, px(pose[i]), 3, (50, 255, 150), -1)

    for hand, color_line, color_dot in [
        (kp["left_hand"],  (255, 130, 0),  (255, 180, 50)),
        (kp["right_hand"], (0,  130, 255), (50,  180, 255)),
    ]:
        if not hand.any():
            continue
        for i, j in _HAND_LINKS:
            if hand[i].any() and hand[j].any():
                cv2.line(frame, px(hand[i]), px(hand[j]), color_line, 2)
        for i in range(len(hand)):
            if hand[i].any():
                cv2.circle(frame, px(hand[i]), 4, color_dot, -1)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, idx_to_label, kp_buffer):
    """
    kp_buffer: list of dicts {'pose':(33,3), 'left_hand':(21,3), 'right_hand':(21,3)}
    """
    T = len(kp_buffer)
    flat = np.concatenate([
        np.array([k["pose"]       for k in kp_buffer]).reshape(T, -1),
        np.array([k["left_hand"]  for k in kp_buffer]).reshape(T, -1),
        np.array([k["right_hand"] for k in kp_buffer]).reshape(T, -1),
    ], axis=1).astype(np.float32)

    # Normalize (same as training)
    mean = flat.mean(axis=0, keepdims=True)
    std  = flat.std(axis=0,  keepdims=True) + 1e-8
    flat = (flat - mean) / std

    seq_len = min(T, MAX_SEQ_LEN)
    if T < MAX_SEQ_LEN:
        flat = np.vstack([flat, np.zeros((MAX_SEQ_LEN - T, flat.shape[1]), dtype=np.float32)])
    else:
        flat = flat[:MAX_SEQ_LEN]

    x       = torch.from_numpy(flat).unsqueeze(0)                # (1, T, 225)
    lengths = torch.tensor([seq_len], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, lengths)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top3_idx  = probs.argsort()[::-1][:3]
    top3      = [(idx_to_label[i], float(probs[i])) for i in top3_idx]
    return top3


# ── Drawing helpers ───────────────────────────────────────────────────────────
def put(frame, text, pos, scale=0.75, color=(255,255,255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thickness)


def draw_results(frame, top3):
    h, w = frame.shape[:2]
    label, conf = top3[0]
    color = (0, 220, 0) if conf >= CONF_THRESH else (0, 165, 255)

    # Main prediction
    if conf < CONF_THRESH:
        put(frame, "Uncertain", (20, h - 130), scale=1.4, color=color)
    else:
        put(frame, label.upper(), (20, h - 130), scale=1.8, color=color)
    put(frame, f"Confidence: {conf*100:.1f}%", (20, h - 90), scale=0.7)

    # Confidence bars for top-3
    bar_x, bar_y, bar_h, bar_max_w = 20, h - 68, 14, 200
    for i, (lbl, p) in enumerate(top3):
        y = bar_y + i * 22
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_w, y + bar_h), (60,60,60), -1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + int(bar_max_w * p), y + bar_h), (100,200,100), -1)
        put(frame, f"{lbl}  {p*100:.0f}%", (bar_x + bar_max_w + 8, y + bar_h - 2), scale=0.5, color=(200,200,200), thickness=1)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("Loading PoseTransformer model...")
    model, idx_to_label = load_model()
    num_classes = len(idx_to_label)
    print(f"  {num_classes} classes | {sum(p.numel() for p in model.parameters()):,} params")

    print("Initializing MediaPipe keypoint extractor...")
    extractor = KeypointExtractor()
    print("Ready.\n")
    print("Controls: SPACE = start/stop recording | Q = quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0)")

    cam_fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_needed    = int(cam_fps * RECORD_SECS)

    state            = "idle"   # idle | countdown | recording | processing
    countdown_start  = 0.0
    kp_buffer        = []       # list of kp dicts accumulated during recording
    last_top3        = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        display = frame.copy()
        h, w    = display.shape[:2]

        # Always extract keypoints for live skeleton overlay
        kp = extractor.extract_from_frame(frame)

        # ── State machine ────────────────────────────────────────────────────
        if state == "idle":
            put(display, "Press SPACE to sign", (20, 36), color=(200,200,200))
            if last_top3:
                draw_results(display, last_top3)

        elif state == "countdown":
            elapsed   = time.time() - countdown_start
            remaining = max(0, 3 - int(elapsed))
            if remaining == 0:
                state    = "recording"
                kp_buffer = []
            cv2.rectangle(display, (0,0), (w,h), (0,0,90), 8)
            put(display, f"Get ready...  {remaining if remaining else 'GO!'}",
                (w//2 - 140, h//2), scale=1.4, color=(100,210,255))

        elif state == "recording":
            kp_buffer.append(kp)
            progress = len(kp_buffer) / frames_needed

            # Red border + progress bar
            cv2.rectangle(display, (0,0), (w,h), (0,0,210), 6)
            cv2.rectangle(display, (0, h-8), (w, h), (50,50,50), -1)
            cv2.rectangle(display, (0, h-8), (int(w*progress), h), (0,200,50), -1)
            secs_left = RECORD_SECS * (1 - progress)
            put(display, "Recording...", (20, 36), color=(50,100,255))
            put(display, f"{secs_left:.1f}s", (w-90, 36), color=(50,220,255))

            if len(kp_buffer) >= frames_needed:
                state = "processing"

        elif state == "processing":
            put(display, "Predicting...", (w//2-100, h//2), scale=1.2, color=(255,210,0))
            cv2.imshow("Mouna — Sign Language Demo", display)
            cv2.waitKey(1)

            try:
                last_top3 = run_inference(model, idx_to_label, kp_buffer)
                print(f"Prediction: {last_top3[0][0]}  ({last_top3[0][1]*100:.1f}%)")
                print(f"  Top-3: {[(l, f'{p*100:.1f}%') for l,p in last_top3]}")
            except Exception as e:
                print(f"Inference error: {e}")
                last_top3 = [("error", 0.0)]

            state = "idle"
            continue

        cv2.imshow("Mouna — Sign Language Demo", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" ") and state == "idle":
            state           = "countdown"
            countdown_start = time.time()
            last_top3       = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
