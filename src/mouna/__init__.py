"""
Mouna - Sign Language Recognition System

A baseline sign language recognition system using WLASL dataset,
Azure Blob Storage, and Databricks.
"""

__version__ = "0.1.0"
__author__ = "Vimal Thomas Joseph"

from mouna.utils.config import load_config
from mouna.utils.logging import setup_logger

__all__ = ["load_config", "setup_logger"]
