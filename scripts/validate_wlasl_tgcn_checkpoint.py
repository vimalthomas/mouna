#!/usr/bin/env python3
"""Offline validation of WLASL's official Pose-TGCN checkpoint BEFORE it ever
touches the live mouna-barath app.

mouna-barath's existing _convert_keypoints() node ordering does NOT match
what this checkpoint was trained on: it places hips at positions 9-10 (WLASL
excluded hips entirely) and has no ear slots at all, while the checkpoint
expects eyes at 9-10 and ears at 11-12 — both hands shifted by 2 positions as
a result. This script builds the CORRECT mapping (from WLASL's own
sign_dataset.py: body_pose_exclude confirms which 13 of OpenPose's 25 body
points are kept, in original BODY_25 order) and checks real accuracy on our
own labeled MediaPipe data (never seen by this checkpoint during training) —
a real integration test, not just an architecture-shape check.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "mouna" / "mouna-barath" / "app" / "backend"))
from models.tgcn_model import GCN_muti_att  # noqa: E402

CKPT_PATH = Path.home() / "mouna" / "mouna" / "data" / "raw" / "wlasl_extra" / "tgcn_pretrained" / "archived" / "asl100" / "ckpt.pth"
ASL100_SPLIT = Path.home() / "mouna" / "mouna" / "data" / "raw" / "wlasl_extra" / "splits" / "splits" / "asl100.json"
RAW_PARQUET = Path.home() / "mouna" / "mouna" / "data" / "silver" / "keypoints" / "keypoints_raw.parquet"

NUM_SAMPLES = 50
NUM_NODES = 55
NUM_CLASSES = 100

# MediaPipe pose landmark indices (within the 33-point pose block)
MP_NOSE, MP_L_EYE, MP_R_EYE = 0, 2, 5
MP_L_SHOULDER, MP_R_SHOULDER = 11, 12
MP_L_ELBOW, MP_R_ELBOW = 13, 14
MP_L_WRIST, MP_R_WRIST = 15, 16
MP_L_HIP, MP_R_HIP = 23, 24
# MediaPipe has no ear landmarks in the 33-point pose set — WLASL's REar/LEar
# slots get the nearest available substitute (eye position) rather than zero,
# since zero would read as "detected at image origin," which is worse than a
# near-miss approximation for two of fifty-five nodes.


def mediapipe_to_wlasl55(flat_225: np.ndarray) -> np.ndarray:
    """flat_225: (T, 225) raw [0,1]-range MediaPipe coords (33 pose + 21+21 hands, x/y/z).
    Returns (T, 55, 2) in WLASL's official OpenPose-derived node order:
    [Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, MidHip,
     REye, LEye, REar(approx), LEar(approx), <21 left hand>, <21 right hand>]
    x,y only (this checkpoint trained on 2D OpenPose data, z unavailable)."""
    T = flat_225.shape[0]
    arr = flat_225.reshape(T, 75, 3)
    pose, lh, rh = arr[:, :33], arr[:, 33:54], arr[:, 54:75]

    out = np.zeros((T, NUM_NODES, 2), dtype=np.float32)
    out[:, 0] = pose[:, MP_NOSE, :2]
    out[:, 1] = (pose[:, MP_L_SHOULDER, :2] + pose[:, MP_R_SHOULDER, :2]) / 2
    out[:, 2] = pose[:, MP_R_SHOULDER, :2]
    out[:, 3] = pose[:, MP_R_ELBOW, :2]
    out[:, 4] = pose[:, MP_R_WRIST, :2]
    out[:, 5] = pose[:, MP_L_SHOULDER, :2]
    out[:, 6] = pose[:, MP_L_ELBOW, :2]
    out[:, 7] = pose[:, MP_L_WRIST, :2]
    out[:, 8] = (pose[:, MP_L_HIP, :2] + pose[:, MP_R_HIP, :2]) / 2
    out[:, 9] = pose[:, MP_R_EYE, :2]
    out[:, 10] = pose[:, MP_L_EYE, :2]
    out[:, 11] = pose[:, MP_R_EYE, :2]  # ear approx
    out[:, 12] = pose[:, MP_L_EYE, :2]  # ear approx
    out[:, 13:34] = lh[:, :, :2]
    out[:, 34:55] = rh[:, :, :2]
    return out


def sample_and_featurize(seq_55_2: np.ndarray) -> np.ndarray:
    """(T, 55, 2) -> (55, NUM_SAMPLES*2) matching GCN_muti_att's expected input:
    NUM_SAMPLES evenly-spaced frames, [x-across-time, y-across-time] per node,
    values normalized to roughly [-1, 1] (OpenPose training used raw-pixel/256
    scaling to this range; MediaPipe is already 0-1 normalized by image size,
    so the equivalent transform is 2*(v-0.5))."""
    T = seq_55_2.shape[0]
    if T >= NUM_SAMPLES:
        idx = np.linspace(0, T - 1, NUM_SAMPLES).astype(int)
    else:
        idx = np.concatenate([np.arange(T), np.full(NUM_SAMPLES - T, T - 1)])
    sampled = seq_55_2[idx]  # (NUM_SAMPLES, 55, 2)
    sampled = 2.0 * (sampled - 0.5)
    x = sampled[:, :, 0].T  # (55, NUM_SAMPLES)
    y = sampled[:, :, 1].T
    return np.concatenate([x, y], axis=1)  # (55, NUM_SAMPLES*2)


def main():
    asl100 = json.load(open(ASL100_SPLIT))
    labels = sorted(g["gloss"] for g in asl100)  # WLASL's own training convention: sorted()
    label_to_idx = {g: i for i, g in enumerate(labels)}
    print(f"Labels: {len(labels)} (sorted, matching WLASL training convention)")

    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = GCN_muti_att(input_feature=NUM_SAMPLES * 2, hidden_feature=64,
                          num_class=NUM_CLASSES, p_dropout=0.3, num_stage=20)
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded cleanly (strict=True, no missing/unexpected keys).")

    import pandas as pd
    df = pd.read_parquet(RAW_PARQUET)
    df = df[df["success"]].reset_index(drop=True)
    sample = df.sample(min(200, len(df)), random_state=1)

    correct_top1, correct_top5, n = 0, 0, 0
    for _, row in sample.iterrows():
        gloss = row["gloss"]
        if gloss not in label_to_idx:
            continue
        flat = pickle.loads(row["keypoints_pickle"])
        mapped = mediapipe_to_wlasl55(flat)
        feat = sample_and_featurize(mapped)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, 55, 100)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[0]
            top5 = torch.topk(probs, 5).indices.tolist()
        true_idx = label_to_idx[gloss]
        n += 1
        if top5[0] == true_idx:
            correct_top1 += 1
        if true_idx in top5:
            correct_top5 += 1

    print(f"\nSanity check on {n} of our own real labeled examples (checkpoint never trained on these):")
    print(f"  top-1 accuracy: {100*correct_top1/n:.1f}%")
    print(f"  top-5 accuracy: {100*correct_top5/n:.1f}%")
    print(f"  (naive chance: top-1 ~1%, top-5 ~5%)")


if __name__ == "__main__":
    main()
