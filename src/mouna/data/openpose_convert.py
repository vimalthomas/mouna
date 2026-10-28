"""Convert WLASL's released OpenPose BODY_25 keypoints to our MediaPipe-shaped
225-dim feature format.

Shared by scripts/convert_openpose_keypoints.py (labeled, 100-sign recovery)
and scripts/build_pretrain_pool.py (unlabeled, full ~21K-video pretraining
pool) — kept here as pure functions with no pandas/pickle/IO dependency so
both callers can reuse the exact same conversion and normalization logic.

Mapping notes (see WHY in each block):
  - Pose: OpenPose BODY_25 has 25 joints, MediaPipe Pose has 33. Only the
    anatomically-shared joints (nose/eyes/ears/shoulders/elbows/wrists/hips/
    knees/ankles/heels/toes) are mapped; MediaPipe-only points with no BODY_25
    equivalent (eye corners, mouth corners, pose-level finger landmarks) are
    left zero.
  - Hands: OpenPose's 21-point hand topology (wrist + 4 joints x 4 fingers) is
    the same convention MediaPipe Hands uses, so hand landmarks map 1:1 with
    no reordering.
  - Depth: OpenPose is 2D-only (x, y, confidence). z is set to 0 for all
    converted points — a real information loss vs. true MediaPipe z.
  - Confidence floor 0.1: measured on a random sample of this data, 91.5% of
    hand points and 53.7% of pose points fall below this — most "detections"
    under it are noise, not signal.
"""

import json
from pathlib import Path

import numpy as np

# OpenPose BODY_25 index -> MediaPipe Pose (33) index. Points with no shared
# anatomical meaning (OpenPose Neck/MidHip, MediaPipe eye-corners/mouth/pose
# fingertips) are intentionally absent from this map and stay zero-filled.
BODY25_TO_MEDIAPIPE = {
    0: 0,    # Nose -> nose
    15: 5,   # REye -> right_eye
    16: 2,   # LEye -> left_eye
    17: 8,   # REar -> right_ear
    18: 7,   # LEar -> left_ear
    2: 12,   # RShoulder -> right_shoulder
    5: 11,   # LShoulder -> left_shoulder
    3: 14,   # RElbow -> right_elbow
    6: 13,   # LElbow -> left_elbow
    4: 16,   # RWrist -> right_wrist
    7: 15,   # LWrist -> left_wrist
    9: 24,   # RHip -> right_hip
    12: 23,  # LHip -> left_hip
    10: 26,  # RKnee -> right_knee
    13: 25,  # LKnee -> left_knee
    11: 28,  # RAnkle -> right_ankle
    14: 27,  # LAnkle -> left_ankle
    24: 30,  # RHeel -> right_heel
    21: 29,  # LHeel -> left_heel
    22: 32,  # RBigToe -> right_foot_index (approx)
    19: 31,  # LBigToe -> left_foot_index (approx)
}

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
CONF_FLOOR = 0.1
MAX_FRAMES = 150

POSE_DIR = Path("data/raw/wlasl_extra/pose_keypoints/pose_per_individual_videos")


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Mirrors KeypointExtractor.normalize_keypoints exactly (per-sequence z-score),
    duplicated here to avoid importing mouna.data.preprocessing's cv2/mediapipe
    dependency chain for what is otherwise a pure-numpy function."""
    mean = np.mean(keypoints, axis=0, keepdims=True)
    centered = keypoints - mean
    std = np.std(centered, axis=0, keepdims=True) + 1e-8
    return centered / std


def _triplets(flat_list):
    arr = np.array(flat_list, dtype=np.float32).reshape(-1, 3)
    return arr  # (N, 3) as [x, y, confidence]


def _convert_frame(op_json: dict) -> dict:
    pose_kp = np.zeros((POSE_LANDMARKS, 3), dtype=np.float32)
    left_hand_kp = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
    right_hand_kp = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)

    people = op_json.get("people", [])
    if not people:
        return {"pose": pose_kp, "left_hand": left_hand_kp, "right_hand": right_hand_kp}

    person = people[0]

    body = _triplets(person.get("pose_keypoints_2d", []))
    for op_idx, mp_idx in BODY25_TO_MEDIAPIPE.items():
        if op_idx < len(body):
            x, y, conf = body[op_idx]
            if conf > CONF_FLOOR:
                pose_kp[mp_idx] = [x, y, 0.0]  # z unavailable from OpenPose 2D

    lh = _triplets(person.get("hand_left_keypoints_2d", []))
    if len(lh) == HAND_LANDMARKS:
        mask = lh[:, 2] > CONF_FLOOR
        left_hand_kp[mask, :2] = lh[mask, :2]

    rh = _triplets(person.get("hand_right_keypoints_2d", []))
    if len(rh) == HAND_LANDMARKS:
        mask = rh[:, 2] > CONF_FLOOR
        right_hand_kp[mask, :2] = rh[mask, :2]

    return {"pose": pose_kp, "left_hand": left_hand_kp, "right_hand": right_hand_kp}


def convert_video(video_id: str, max_frames: int = MAX_FRAMES, pose_dir: Path = POSE_DIR) -> np.ndarray:
    folder = pose_dir / video_id
    frame_files = sorted(folder.glob("*_keypoints.json"))[:max_frames]
    if not frame_files:
        raise ValueError("no keypoint frames found")

    pose_seq, left_seq, right_seq = [], [], []
    for ff in frame_files:
        with open(ff) as f:
            op_json = json.load(f)
        kp = _convert_frame(op_json)
        pose_seq.append(kp["pose"])
        left_seq.append(kp["left_hand"])
        right_seq.append(kp["right_hand"])

    pose = np.array(pose_seq)
    left = np.array(left_seq)
    right = np.array(right_seq)

    pose_flat = pose.reshape(pose.shape[0], -1)
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    return np.concatenate([pose_flat, left_flat, right_flat], axis=1)  # (T, 225)
