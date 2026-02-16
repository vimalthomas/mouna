"""Video preprocessing and keypoint extraction using MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from tqdm import tqdm


class KeypointExtractor:
    """Extract pose, hand, and face keypoints from videos using MediaPipe."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Holistic model.

        Args:
            static_image_mode: Whether to treat images as static (vs video stream).
            model_complexity: Complexity of pose model (0, 1, or 2).
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
        """
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract keypoints from a single frame.

        Args:
            frame: RGB image frame.

        Returns:
            Dictionary with pose, left_hand, right_hand, and face keypoints.
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.holistic.process(frame_rgb)

        keypoints = {
            "pose": self._landmarks_to_array(results.pose_landmarks, 33),
            "left_hand": self._landmarks_to_array(results.left_hand_landmarks, 21),
            "right_hand": self._landmarks_to_array(results.right_hand_landmarks, 21),
            "face": self._landmarks_to_array(results.face_landmarks, 468),
        }

        return keypoints

    def extract_from_video(
        self, video_path: str, max_frames: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract keypoints from entire video.

        Args:
            video_path: Path to video file.
            max_frames: Maximum number of frames to process.

        Returns:
            Dictionary with temporal sequences of keypoints.
            Shape: (num_frames, num_keypoints, 3) for x, y, z coordinates.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        # Storage for temporal sequences
        pose_seq = []
        left_hand_seq = []
        right_hand_seq = []
        face_seq = []

        frame_idx = 0
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = self.extract_from_frame(frame)

            pose_seq.append(keypoints["pose"])
            left_hand_seq.append(keypoints["left_hand"])
            right_hand_seq.append(keypoints["right_hand"])
            face_seq.append(keypoints["face"])

            frame_idx += 1

        cap.release()

        return {
            "pose": np.array(pose_seq),  # (T, 33, 3)
            "left_hand": np.array(left_hand_seq),  # (T, 21, 3)
            "right_hand": np.array(right_hand_seq),  # (T, 21, 3)
            "face": np.array(face_seq),  # (T, 468, 3)
        }

    def _landmarks_to_array(
        self, landmarks, expected_size: int
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks to numpy array.

        Args:
            landmarks: MediaPipe landmarks object.
            expected_size: Expected number of landmarks.

        Returns:
            Array of shape (expected_size, 3) with x, y, z coordinates.
            Returns zeros if landmarks not detected.
        """
        if landmarks is None:
            return np.zeros((expected_size, 3))

        coords = []
        for landmark in landmarks.landmark:
            coords.append([landmark.x, landmark.y, landmark.z])

        return np.array(coords)

    def flatten_keypoints(self, keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten all keypoints into single feature vector per frame.

        Args:
            keypoints: Dictionary with pose, hands, and face keypoints.

        Returns:
            Flattened array of shape (num_frames, 543*3) = (num_frames, 1629).
            543 = 33 (pose) + 21 (left_hand) + 21 (right_hand) + 468 (face)
        """
        pose = keypoints["pose"]  # (T, 33, 3)
        left_hand = keypoints["left_hand"]  # (T, 21, 3)
        right_hand = keypoints["right_hand"]  # (T, 21, 3)
        face = keypoints["face"]  # (T, 468, 3)

        # Flatten spatial dimensions, keep temporal
        pose_flat = pose.reshape(pose.shape[0], -1)  # (T, 99)
        left_flat = left_hand.reshape(left_hand.shape[0], -1)  # (T, 63)
        right_flat = right_hand.reshape(right_hand.shape[0], -1)  # (T, 63)
        face_flat = face.reshape(face.shape[0], -1)  # (T, 1404)

        # Concatenate all features
        all_features = np.concatenate(
            [pose_flat, left_flat, right_flat, face_flat], axis=1
        )  # (T, 1629)

        return all_features

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints for translation and scale invariance.

        Args:
            keypoints: Array of shape (num_frames, num_features).

        Returns:
            Normalized keypoints.
        """
        # Center around mean
        mean = np.mean(keypoints, axis=0, keepdims=True)
        centered = keypoints - mean

        # Scale to unit variance
        std = np.std(centered, axis=0, keepdims=True) + 1e-8
        normalized = centered / std

        return normalized


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

        # Calculate frame sampling rate
        sample_rate = max(1, int(original_fps / self.target_fps))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames based on target FPS
            if frame_idx % sample_rate == 0:
                # Resize
                frame = cv2.resize(frame, self.target_size)

                # Normalize if requested
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
            # Truncate
            return sequence[:max_length]
        else:
            # Pad
            pad_shape = (max_length - current_length,) + sequence.shape[1:]
            padding = np.full(pad_shape, pad_value, dtype=sequence.dtype)
            return np.concatenate([sequence, padding], axis=0)
