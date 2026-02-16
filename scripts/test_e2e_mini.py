#!/usr/bin/env python3
"""
End-to-end mini test with synthetic data.
Tests the entire pipeline without needing real WLASL data.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from mouna.data.preprocessing import KeypointExtractor
from mouna.models.baseline import KeypointBiLSTM
from mouna.utils.logging import setup_logger


def create_synthetic_video(output_path: str, num_frames: int = 30):
    """Create a synthetic video for testing."""
    height, width = 480, 640
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a frame with moving circle (simulating hand movement)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Moving circle
        center_x = int(width / 2 + 100 * np.sin(i * 0.2))
        center_y = int(height / 2 + 100 * np.cos(i * 0.2))
        cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), -1)

        out.write(frame)

    out.release()
    print(f"Created synthetic video: {output_path}")


def test_keypoint_extraction():
    """Test keypoint extraction on synthetic video."""
    print("\n" + "="*60)
    print("Test 1: Keypoint Extraction")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_video.mp4")
        create_synthetic_video(video_path, num_frames=30)

        # Extract keypoints
        extractor = KeypointExtractor()
        keypoints = extractor.extract_from_video(video_path)

        print(f"✓ Extracted keypoints from video")
        print(f"  Pose shape: {keypoints['pose'].shape}")
        print(f"  Left hand shape: {keypoints['left_hand'].shape}")
        print(f"  Right hand shape: {keypoints['right_hand'].shape}")
        print(f"  Face shape: {keypoints['face'].shape}")

        # Flatten
        flattened = extractor.flatten_keypoints(keypoints)
        print(f"✓ Flattened keypoints: {flattened.shape}")

        # Normalize
        normalized = extractor.normalize_keypoints(flattened)
        print(f"✓ Normalized keypoints: {normalized.shape}")

        return normalized


def test_model_training():
    """Test model training with synthetic data."""
    print("\n" + "="*60)
    print("Test 2: Model Training (5 epochs)")
    print("="*60)

    # Create synthetic dataset
    num_samples = 20
    num_classes = 5
    seq_len = 50
    feature_dim = 1629

    print(f"Creating synthetic dataset: {num_samples} samples, {num_classes} classes")

    # Synthetic data
    X_train = torch.randn(num_samples, seq_len, feature_dim)
    y_train = torch.randint(0, num_classes, (num_samples,))
    lengths = torch.randint(30, seq_len, (num_samples,))

    # Create simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, lengths):
            self.X = X
            self.y = y
            self.lengths = lengths

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return {
                'keypoints': self.X[idx],
                'label': self.y[idx],
                'sequence_length': self.lengths[idx]
            }

    dataset = SimpleDataset(X_train, y_train, lengths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model
    model = KeypointBiLSTM(
        input_dim=feature_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"✓ Model initialized")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            keypoints = batch['keypoints'].to(device)
            labels = batch['label'].to(device)
            seq_lengths = batch['sequence_length'].to(device)

            # Forward pass
            outputs = model(keypoints, seq_lengths)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)

        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    print(f"✓ Training completed successfully")

    return model


def test_evaluation(model):
    """Test evaluation metrics."""
    print("\n" + "="*60)
    print("Test 3: Model Evaluation")
    print("="*60)

    # Create test data
    num_samples = 10
    num_classes = 5
    seq_len = 50
    feature_dim = 1629

    X_test = torch.randn(num_samples, seq_len, feature_dim)
    y_test = torch.randint(0, num_classes, (num_samples,))
    lengths = torch.full((num_samples,), seq_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(X_test.to(device), lengths.to(device))
        predictions = outputs.argmax(dim=1).cpu()

    # Calculate metrics
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test.numpy(), predictions.numpy())

    print(f"✓ Evaluation completed")
    print(f"  Test accuracy: {accuracy*100:.2f}%")
    print(f"  Predictions: {predictions.tolist()}")
    print(f"  Ground truth: {y_test.tolist()}")

    return accuracy


def test_model_save_load():
    """Test model checkpointing."""
    print("\n" + "="*60)
    print("Test 4: Model Save/Load")
    print("="*60)

    model = KeypointBiLSTM(
        input_dim=1629,
        hidden_dim=128,
        num_layers=2,
        num_classes=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "model.pth")

        # Save
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_dim': 1629,
                'hidden_dim': 128,
                'num_layers': 2,
                'num_classes': 5,
            }
        }, checkpoint_path)

        print(f"✓ Model saved to {checkpoint_path}")

        # Load
        checkpoint = torch.load(checkpoint_path)
        new_model = KeypointBiLSTM(**checkpoint['config'])
        new_model.load_state_dict(checkpoint['model_state_dict'])

        print(f"✓ Model loaded successfully")

        # Verify
        test_input = torch.randn(1, 50, 1629)
        test_lengths = torch.tensor([50])

        model.eval()
        new_model.eval()

        with torch.no_grad():
            out1 = model(test_input, test_lengths)
            out2 = new_model(test_input, test_lengths)

        assert torch.allclose(out1, out2, atol=1e-6)
        print(f"✓ Model outputs match after save/load")


def main():
    """Run all end-to-end tests."""
    setup_logger(log_level="INFO")

    print("="*60)
    print("Mouna End-to-End Mini Test")
    print("Testing with synthetic data (no WLASL needed)")
    print("="*60)

    try:
        # Test 1: Keypoint extraction
        test_keypoint_extraction()

        # Test 2: Model training
        model = test_model_training()

        # Test 3: Evaluation
        test_evaluation(model)

        # Test 4: Save/load
        test_model_save_load()

        # Summary
        print("\n" + "="*60)
        print("✓ All end-to-end tests passed!")
        print("="*60)
        print("\nYour pipeline is working correctly.")
        print("Next steps:")
        print("  1. Download WLASL data: make download-wlasl")
        print("  2. Train on real data: make train-baseline")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
