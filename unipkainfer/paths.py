"""Filesystem locations owned by unipkainfer."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_path


def default_model_dir() -> Path:
    """Return the default per-user directory for downloaded model checkpoints."""

    return Path(user_data_path("unipkainfer", appauthor=False)) / "models"
