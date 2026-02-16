"""Configuration management using Pydantic."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class AzureConfig(BaseSettings):
    """Azure configuration."""

    subscription_id: str = Field(..., env="AZURE_SUBSCRIPTION_ID")
    resource_group: str = Field(..., env="AZURE_RESOURCE_GROUP")
    storage_account: str = Field(..., env="AZURE_STORAGE_ACCOUNT")
    region: str = Field(default="eastus", env="AZURE_REGION")
    storage_key: str = Field(..., env="AZURE_STORAGE_KEY")

    @property
    def connection_string(self) -> str:
        """Generate Azure Storage connection string."""
        return (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={self.storage_account};"
            f"AccountKey={self.storage_key};"
            f"EndpointSuffix=core.windows.net"
        )


class DataConfig(BaseSettings):
    """Data configuration."""

    azure_container_bronze: str = "sign-videos-bronze"
    azure_container_silver: str = "sign-videos-silver"
    azure_container_gold: str = "sign-videos-gold"
    num_classes: int = 2000  # WLASL-2000
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


class BaselineModelConfig(BaseSettings):
    """Baseline model configuration."""

    input_dim: int = 1662  # MediaPipe: 543 landmarks * 3 coordinates (x, y, z)
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    model_type: str = "bilstm"  # bilstm or gru


class MultimodalModelConfig(BaseSettings):
    """Multimodal model configuration."""

    vision_backbone: str = "timesformer"  # timesformer, videomae, i3d
    temporal_model: str = "transformer"  # transformer, lstm, gru
    fusion_strategy: str = "cross_attention"  # late_fusion, cross_attention
    hidden_dim: int = 768
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1


class TrainingConfig(BaseSettings):
    """Training configuration."""

    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    num_workers: int = 4
    device: str = "cuda"  # cuda or cpu


class GeminiConfig(BaseSettings):
    """Gemini QA configuration."""

    api_key: str = Field(..., env="GEMINI_API_KEY")
    model: str = "gemini-1.5-pro"
    confidence_threshold: float = 0.7
    max_retries: int = 3


class MLflowConfig(BaseSettings):
    """MLflow configuration."""

    tracking_uri: str = Field(default="./mlruns", env="MLFLOW_TRACKING_URI")
    experiment_name: str = "sign-recognition-baseline"
    run_name: Optional[str] = None


class Config(BaseSettings):
    """Main configuration object."""

    azure: AzureConfig = Field(default_factory=AzureConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    baseline: BaselineModelConfig = Field(default_factory=BaselineModelConfig)
    multimodal: MultimodalModelConfig = Field(default_factory=MultimodalModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to YAML configuration file. If None, uses default config.

    Returns:
        Config object with all settings.
    """
    # Load base config
    config_dict: Dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

    # Override with environment variables
    return Config(**config_dict)


def save_config(config: Config, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save.
        output_path: Path to save YAML file.
    """
    config_dict = config.model_dump()

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
