"""Shared test setup: make the backend and core packages importable, and give
tests the repo paths. Backend modules read config relative to app/backend,
so tests run from there; data paths are absolute (ROOT)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
sys.path[:0] = [str(BACKEND), str(ROOT / "src"), str(ROOT / "scripts")]
os.chdir(BACKEND)


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(autouse=True)
def no_ai_keys(monkeypatch):
    """Tests check the app's own logic and never call a paid AI API, even
    when a key is in app/backend/.env."""
    from config import settings
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "")
