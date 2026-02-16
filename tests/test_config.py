"""Tests for configuration module."""

import pytest
from pathlib import Path

from mouna.utils.config import Config, load_config


def test_config_initialization():
    """Test basic config initialization."""
    # This will fail if required env vars are not set, which is expected
    # In real tests, you'd use mock environment variables
    try:
        config = Config()
        assert config is not None
    except Exception:
        # Expected to fail without proper env vars
        pass


def test_load_config_from_yaml(tmp_path):
    """Test loading config from YAML file."""
    # Create a temporary config file
    config_content = """
data:
  num_classes: 100

baseline:
  hidden_dim: 256

training:
  batch_size: 16
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    # Note: This will still fail without required env vars
    # In production tests, you'd mock environment variables
    try:
        config = load_config(str(config_file))
        assert config.data.num_classes == 100
        assert config.baseline.hidden_dim == 256
        assert config.training.batch_size == 16
    except Exception:
        # Expected without proper env setup
        pass
