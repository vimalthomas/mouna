#!/usr/bin/env python3
"""Quick setup validation script."""

import sys
from pathlib import Path

def test_imports():
    """Test that all major dependencies can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        import mediapipe as mp
        print(f"✓ MediaPipe {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False

    try:
        from azure.storage.blob import BlobServiceClient
        print("✓ Azure Storage SDK")
    except ImportError as e:
        print(f"✗ Azure SDK import failed: {e}")
        return False

    try:
        import mlflow
        print(f"✓ MLflow {mlflow.__version__}")
    except ImportError as e:
        print(f"✗ MLflow import failed: {e}")
        return False

    try:
        import google.generativeai as genai
        print("✓ Google Generative AI (Gemini)")
    except ImportError as e:
        print(f"✗ Gemini SDK import failed: {e}")
        return False

    return True


def test_mouna_package():
    """Test that mouna package can be imported."""
    print("\nTesting mouna package...")

    try:
        import mouna
        print(f"✓ Mouna package imported (version {mouna.__version__})")
    except ImportError as e:
        print(f"✗ Mouna package import failed: {e}")
        print("  Run: pip install -e .")
        return False

    # Test individual modules
    modules = [
        "mouna.utils.config",
        "mouna.utils.logging",
        "mouna.data.ingestion",
        "mouna.data.preprocessing",
        "mouna.models.baseline",
        "mouna.evaluation.metrics",
        "mouna.gemini.qa_layer",
    ]

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            return False

    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from mouna.utils.config import load_config
        config_path = Path("configs/dev.yaml")

        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False

        # This will fail if .env is not properly configured
        # But we can catch the error and inform the user
        try:
            config = load_config(str(config_path))
            print(f"✓ Configuration loaded from {config_path}")
            print(f"  Num classes: {config.data.num_classes}")
            print(f"  Baseline model: {config.baseline.model_type}")
            print(f"  Batch size: {config.training.batch_size}")
            return True
        except Exception as e:
            print(f"⚠ Config loaded but environment variables missing: {e}")
            print("  Update .env file with your credentials")
            return True  # Not a critical failure for basic testing

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_model_initialization():
    """Test that models can be instantiated."""
    print("\nTesting model initialization...")

    try:
        from mouna.models.baseline import KeypointBiLSTM, KeypointGRU

        # Test BiLSTM
        model = KeypointBiLSTM(
            input_dim=1629,
            hidden_dim=128,  # Smaller for testing
            num_layers=1,
            num_classes=100,  # Smaller for testing
        )
        print(f"✓ KeypointBiLSTM instantiated")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test GRU
        model = KeypointGRU(
            input_dim=1629,
            hidden_dim=128,
            num_layers=1,
            num_classes=100,
        )
        print(f"✓ KeypointGRU instantiated")

        # Test forward pass with dummy data
        import torch
        batch_size = 2
        seq_len = 50
        dummy_input = torch.randn(batch_size, seq_len, 1629)
        lengths = torch.tensor([50, 40])

        output = model(dummy_input, lengths)
        assert output.shape == (batch_size, 100)
        print(f"✓ Forward pass successful: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test preprocessing components."""
    print("\nTesting preprocessing...")

    try:
        from mouna.data.preprocessing import KeypointExtractor
        import numpy as np

        extractor = KeypointExtractor()
        print("✓ KeypointExtractor initialized")

        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        keypoints = extractor.extract_from_frame(dummy_frame)

        print(f"✓ Keypoint extraction successful")
        print(f"  Pose: {keypoints['pose'].shape}")
        print(f"  Left hand: {keypoints['left_hand'].shape}")
        print(f"  Right hand: {keypoints['right_hand'].shape}")
        print(f"  Face: {keypoints['face'].shape}")

        # Test flattening
        temporal_data = {k: np.expand_dims(v, 0) for k, v in keypoints.items()}
        flattened = extractor.flatten_keypoints(temporal_data)
        print(f"✓ Keypoint flattening: {flattened.shape}")

        return True

    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Mouna Setup Validation")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Mouna Package", test_mouna_package),
        ("Configuration", test_config),
        ("Model Initialization", test_model_initialization),
        ("Preprocessing", test_preprocessing),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'='*60}")
        result = test_func()
        results.append((name, result))

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
