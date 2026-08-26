"""Configuration models and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SurgiGuardConfig:
    threshold: float = 0.5
    history_size: int = 5
    min_confidence: float = 0.5
    trim_fraction: float = 0.2
    disagreement_threshold: float = 0.2
    change_threshold: float = 0.2
    trend_window: int = 8
    trend_horizon_seconds: float = 1.0
    alert_threshold: float = 0.05
    alert_persistence: int = 3
    cooldown_seconds: float = 5.0


def load_config(path: str | Path) -> SurgiGuardConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return SurgiGuardConfig(**values)

