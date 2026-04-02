"""Load YAML configuration and expose typed accessors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils import project_root


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ProjectConfig:
    """Minimal wrapper; full dict kept for flexibility."""

    raw: dict[str, Any]
    config_path: Path

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_seed"])

    @property
    def horizon_days(self) -> int:
        return int(self.raw["project"]["horizon_days"])

    @property
    def output_dir(self) -> Path:
        root = project_root()
        return root / self.raw["paths"]["output_dir"]

    @property
    def reports_dir(self) -> Path:
        root = project_root()
        return root / self.raw["paths"]["reports_dir"]


def load_config(config_path: Path | str | None = None) -> ProjectConfig:
    root = project_root()
    path = Path(config_path) if config_path else root / "configs" / "default.yaml"
    raw = load_yaml(path)
    return ProjectConfig(raw=raw, config_path=path.resolve())
