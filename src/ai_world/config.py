"""Global configuration: data paths and hard limits."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "ai-world"

# --- Map limits ---------------------------------------------------------
MIN_MAP_DIM = 16
MAX_MAP_DIM = 4096
DEFAULT_MAP_DIM = 256

# --- Window ------------------------------------------------------------
DEFAULT_WINDOW_SIZE = (1280, 800)
TARGET_FPS = 60


def data_dir() -> Path:
    """User data directory (world database). Overridable via AI_WORLD_DATA_DIR."""
    override = os.environ.get("AI_WORLD_DATA_DIR")
    if override:
        path = Path(override).expanduser()
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        path = Path(base) / APP_NAME
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
        path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return data_dir() / "worlds.db"
