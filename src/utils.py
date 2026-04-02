"""Logging, random seeds, and artifact paths."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for CLI and pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_global_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility (best-effort across libraries)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import catboost as cb  # noqa: F401

        # CatBoost uses random_seed in train; global numpy seed still helps sklearn split
    except ImportError:
        pass


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path | str, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def project_root() -> Path:
    """Root of repository (parent of `src/`)."""
    return Path(__file__).resolve().parent.parent
