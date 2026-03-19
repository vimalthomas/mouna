"""Video preprocessing and keypoint extraction using MediaPipe Tasks API."""

import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Model URLs from MediaPipe Model Hub
_MODEL_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
_MODEL_CACHE_DIR = Path.home() / ".mouna" / "models"

# Feature dimensions: 33 pose + 21 left_hand + 21 right_hand (no face)
# Each landmark has x, y, z = 3 values
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
KEYPOINT_DIM = (POSE_LANDMARKS + HAND_LANDMARKS * 2) * 3  # 225


def _get_model_path(name: str) -> str:
    """Download model file if not cached, return local path."""
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = _MODEL_URLS[name].split("/")[-1]
    local_path = _MODEL_CACHE_DIR / filename
    if not local_path.exists():
        logger.info(f"Downloading MediaPipe {name} model to {local_path}...")
        urllib.request.urlretrieve(_MODEL_URLS[name], local_path)
        logger.info(f"Downloaded {name} model.")
    return str(local_path)


class KeypointExtractor:
    """Extract pose and hand keypoints from videos using MediaPipe Tasks API."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Pose and Hand landmarkers.

        Args:
            min_detection_confidence: Minimum confidence for detection.
        """
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_get_model_path("pose")),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
        )
        self._pose = vision.PoseLandmarker.create_from_options(pose_opts)

        hand_opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_get_model_path("hand")),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
        )
        self._hand = vision.HandLandmarker.create_from_options(hand_opts)

    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract keypoints from a single frame.

        Args:
            frame: BGR image frame (as from OpenCV).

        Returns:
            Dictionary with pose, left_hand, right_hand keypoints.
            Each value has shape (N, 3) for x, y, z coordinates.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self._pose.detect(mp_image)
        hand_result = self._hand.detect(mp_image)

        pose_kp = np.zeros((POSE_LANDMARKS, 3))
        if pose_result.pose_landmarks:
            for i, lm in enumerate(pose_result.pose_landmarks[0]):
                pose_kp[i] = [lm.x, lm.y, lm.z]

        left_hand_kp = np.zeros((HAND_LANDMARKS, 3))
        right_hand_kp = np.zeros((HAND_LANDMARKS, 3))
        if hand_result.hand_landmarks:
            for landmarks, handedness in zip(
                hand_result.hand_landmarks, hand_result.handedness
            ):
                label = handedness[0].category_name  # "Left" or "Right"
                kp = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                if label == "Left":
                    left_hand_kp = kp
                else:
                    right_hand_kp = kp

        return {
            "pose": pose_kp,
            "left_hand": left_hand_kp,
            "right_hand": right_hand_kp,
        }

    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        end_frame: int = -1,
    ) -> Dict[str, np.ndarray]:
        """
        Extract keypoints from a video, optionally restricted to a frame range.

        Args:
            video_path:   Path to video file.
            max_frames:   Maximum number of frames to extract (cap on output length).
            start_frame:  First frame index to extract (0-based). Use 0 for start of file.
            end_frame:    Last frame index to extract (inclusive). Use -1 for end of file.

        Returns:
            Dictionary with temporal sequences of keypoints.
            Shape: (num_frames, num_landmarks, 3).
        """
        cap = cv2.VideoCapture(video_path)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        pose_seq, left_hand_seq, right_hand_seq = [], [], []
        frame_idx = start_frame

        while cap.isOpened():
            if max_frames is not None and len(pose_seq) >= max_frames:
                break
            if end_frame >= 0 and frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            kp = self.extract_from_frame(frame)
            pose_seq.append(kp["pose"])
            left_hand_seq.append(kp["left_hand"])
            right_hand_seq.append(kp["right_hand"])
            frame_idx += 1

        cap.release()

        return {
            "pose": np.array(pose_seq),        # (T, 33, 3)
            "left_hand": np.array(left_hand_seq),  # (T, 21, 3)
            "right_hand": np.array(right_hand_seq), # (T, 21, 3)
        }


    def flatten_keypoints(self, keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten all keypoints into single feature vector per frame.

        Args:
            keypoints: Dictionary with pose, left_hand, right_hand keypoints.

        Returns:
            Flattened array of shape (num_frames, 225).
            225 = (33 pose + 21 left_hand + 21 right_hand) * 3
        """
        pose = keypoints["pose"]            # (T, 33, 3)
        left_hand = keypoints["left_hand"]  # (T, 21, 3)
        right_hand = keypoints["right_hand"] # (T, 21, 3)

        pose_flat = pose.reshape(pose.shape[0], -1)             # (T, 99)
        left_flat = left_hand.reshape(left_hand.shape[0], -1)  # (T, 63)
        right_flat = right_hand.reshape(right_hand.shape[0], -1) # (T, 63)

        return np.concatenate([pose_flat, left_flat, right_flat], axis=1)  # (T, 225)

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints for translation and scale invariance.

        Args:
            keypoints: Array of shape (num_frames, num_features).

        Returns:
            Normalized keypoints.
        """
        mean = np.mean(keypoints, axis=0, keepdims=True)
        centered = keypoints - mean
        std = np.std(centered, axis=0, keepdims=True) + 1e-8
        return centered / std


class VideoPreprocessor:
    """Preprocess videos for model input."""

    def __init__(
        self,
        target_fps: int = 30,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
    ):
        """
        Initialize video preprocessor.

        Args:
            target_fps: Target frames per second for resampling.
            target_size: Target frame size (height, width).
            normalize: Whether to normalize pixel values to [0, 1].
        """
        self.target_fps = target_fps
        self.target_size = target_size
        self.normalize = normalize

    def preprocess_video(
        self, video_path: str, max_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Load and preprocess video frames.

        Args:
            video_path: Path to video file.
            max_frames: Maximum number of frames to load.

        Returns:
            Preprocessed frames of shape (num_frames, height, width, channels).
        """
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        frame_idx = 0
        sample_rate = max(1, int(original_fps / self.target_fps))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                frame = cv2.resize(frame, self.target_size)
                if self.normalize:
                    frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            frame_idx += 1

            if max_frames and len(frames) >= max_frames:
                break

        cap.release()

        return np.array(frames)

    def temporal_padding(
        self, sequence: np.ndarray, max_length: int, pad_value: float = 0.0
    ) -> np.ndarray:
        """
        Pad or truncate sequence to fixed length.

        Args:
            sequence: Temporal sequence of shape (time, ...).
            max_length: Target sequence length.
            pad_value: Value to use for padding.

        Returns:
            Padded/truncated sequence of shape (max_length, ...).
        """
        current_length = sequence.shape[0]

        if current_length >= max_length:
            return sequence[:max_length]
        else:
            pad_shape = (max_length - current_length,) + sequence.shape[1:]
            padding = np.full(pad_shape, pad_value, dtype=sequence.dtype)
            return np.concatenate([sequence, padding], axis=0)
