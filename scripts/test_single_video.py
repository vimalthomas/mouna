#!/usr/bin/env python3
"""Test with a single video file."""

import sys
from pathlib import Path

from mouna.data.preprocessing import KeypointExtractor
from mouna.models.baseline import KeypointBiLSTM
import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single_video.py <path_to_video.mp4>")
        print("Example: python test_single_video.py ~/Downloads/test_video.mp4")
        return 1

    video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    print(f"Testing with video: {video_path}")

    # Extract keypoints
    print("\n1. Extracting keypoints...")
    extractor = KeypointExtractor()
    keypoints = extractor.extract_from_video(video_path)

    print(f"✓ Keypoints extracted")
    print(f"  Frames: {keypoints['pose'].shape[0]}")
    print(f"  Pose landmarks: {keypoints['pose'].shape}")

    # Flatten and normalize
    flattened = extractor.flatten_keypoints(keypoints)
    normalized = extractor.normalize_keypoints(flattened)
    print(f"✓ Flattened & normalized: {normalized.shape}")

    # Create model
    print("\n2. Initializing model...")
    model = KeypointBiLSTM(
        input_dim=1629,
        hidden_dim=256,
        num_layers=2,
        num_classes=100,
    )
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Inference
    print("\n3. Running inference...")
    model.eval()

    # Prepare input
    keypoints_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    seq_length = torch.tensor([normalized.shape[0]])

    with torch.no_grad():
        output = model(keypoints_tensor, seq_length)
        probabilities = torch.softmax(output, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, k=5, dim=1)

    print(f"✓ Inference successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Top-5 predictions:")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        print(f"    {i+1}. Class {idx.item()}: {prob.item()*100:.2f}%")

    print("\n✓ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
