"""PyTorch dataset for sign language recognition."""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from mouna.data.preprocessing import KeypointExtractor, VideoPreprocessor


class SignLanguageDataset(Dataset):
    """Dataset for sign language videos with keypoints."""

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        split: str = "train",
        max_sequence_length: int = 150,
        use_keypoints: bool = True,
        use_rgb: bool = False,
        transform=None,
    ):
        """
        Initialize sign language dataset.

        Args:
            data_dir: Directory containing video files.
            metadata_path: Path to WLASL metadata JSON.
            split: Dataset split ('train', 'val', 'test').
            max_sequence_length: Maximum sequence length for padding.
            use_keypoints: Whether to extract keypoints.
            use_rgb: Whether to load RGB frames.
            transform: Optional transform to apply.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.use_keypoints = use_keypoints
        self.use_rgb = use_rgb
        self.transform = transform

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Build sample list
        self.samples = self._build_sample_list()

        # Create label mapping
        self.label_to_idx = self._create_label_mapping()
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Initialize preprocessors
        if self.use_keypoints:
            self.keypoint_extractor = KeypointExtractor()
        if self.use_rgb:
            self.video_preprocessor = VideoPreprocessor()

    def _build_sample_list(self) -> List[Dict]:
        """Build list of samples for this split."""
        samples = []

        for gloss_entry in self.metadata:
            gloss = gloss_entry.get("gloss")
            instances = gloss_entry.get("instances", [])

            for instance in instances:
                # Filter by split
                instance_split = instance.get("split", "train")
                if instance_split != self.split:
                    continue

                video_id = instance.get("video_id")
                video_path = self.data_dir / gloss / f"{video_id}.mp4"

                if video_path.exists():
                    samples.append(
                        {
                            "video_path": str(video_path),
                            "gloss": gloss,
                            "video_id": video_id,
                            "signer_id": instance.get("signer_id"),
                        }
                    )

        return samples

    def _create_label_mapping(self) -> Dict[str, int]:
        """Create mapping from gloss to integer label."""
        unique_glosses = sorted(set(entry["gloss"] for entry in self.metadata))
        return {gloss: idx for idx, gloss in enumerate(unique_glosses)}

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - keypoints: (max_seq_len, feature_dim) if use_keypoints
                - rgb_frames: (max_seq_len, H, W, C) if use_rgb
                - label: Integer label
                - sequence_length: Actual sequence length before padding
                - gloss: String gloss label
        """
        sample = self.samples[idx]
        video_path = sample["video_path"]
        gloss = sample["gloss"]
        label = self.label_to_idx[gloss]

        output = {
            "label": torch.tensor(label, dtype=torch.long),
            "gloss": gloss,
            "video_id": sample["video_id"],
            "signer_id": sample.get("signer_id", -1),
        }

        # Extract keypoints
        if self.use_keypoints:
            keypoint_data = self.keypoint_extractor.extract_from_video(video_path)
            keypoints = self.keypoint_extractor.flatten_keypoints(keypoint_data)
            keypoints = self.keypoint_extractor.normalize_keypoints(keypoints)

            # Pad/truncate to max length
            sequence_length = min(len(keypoints), self.max_sequence_length)
            keypoints_padded = self._pad_sequence(keypoints, self.max_sequence_length)

            output["keypoints"] = torch.tensor(keypoints_padded, dtype=torch.float32)
            output["sequence_length"] = torch.tensor(sequence_length, dtype=torch.long)

        # Load RGB frames
        if self.use_rgb:
            frames = self.video_preprocessor.preprocess_video(
                video_path, max_frames=self.max_sequence_length
            )
            sequence_length = min(len(frames), self.max_sequence_length)
            frames_padded = self._pad_sequence(frames, self.max_sequence_length)

            output["rgb_frames"] = torch.tensor(frames_padded, dtype=torch.float32)
            if "sequence_length" not in output:
                output["sequence_length"] = torch.tensor(sequence_length, dtype=torch.long)

        # Apply transforms
        if self.transform:
            output = self.transform(output)

        return output

    def _pad_sequence(
        self, sequence: np.ndarray, max_length: int
    ) -> np.ndarray:
        """Pad or truncate sequence to max_length."""
        current_length = len(sequence)

        if current_length >= max_length:
            return sequence[:max_length]
        else:
            pad_shape = (max_length - current_length,) + sequence.shape[1:]
            padding = np.zeros(pad_shape, dtype=sequence.dtype)
            return np.concatenate([sequence, padding], axis=0)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.

        Returns:
            Tensor of shape (num_classes,) with class weights.
        """
        label_counts = np.zeros(len(self.label_to_idx))

        for sample in self.samples:
            label = self.label_to_idx[sample["gloss"]]
            label_counts[label] += 1

        # Inverse frequency weighting
        total_samples = len(self.samples)
        class_weights = total_samples / (len(self.label_to_idx) * label_counts + 1e-6)

        return torch.tensor(class_weights, dtype=torch.float32)
