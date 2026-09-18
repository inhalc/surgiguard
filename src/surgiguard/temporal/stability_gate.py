"""Evidence-driven stability gate from the KDD formulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _sigmoid(value: FloatArray) -> FloatArray:
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def compute_stability_gate(
    disagreement: FloatArray,
    change_evidence: FloatArray,
    disagreement_threshold: float,
    change_threshold: float,
    disagreement_temperature: float,
    change_temperature: float,
) -> FloatArray:
    """Return a soft-AND gate that is high only under agreement and stability."""

    agreement_term = _sigmoid(
        (disagreement_threshold - np.abs(disagreement)) / disagreement_temperature
    )
    appearance_term = _sigmoid(
        (change_threshold - change_evidence) / change_temperature
    )
    return (agreement_term * appearance_term).astype(np.float32)


def image_change_evidence(
    previous_frame: NDArray[np.uint8],
    current_frame: NDArray[np.uint8],
    low: float,
    high: float,
) -> FloatArray:
    """Normalize aligned appearance change using fixed calibration bounds."""

    previous = previous_frame.astype(np.float32) / 255.0
    current = current_frame.astype(np.float32) / 255.0
    difference = np.abs(current - previous)
    if difference.ndim == 3:
        difference = difference.mean(axis=2)
    return np.clip((difference - low) / (high - low), 0.0, 1.0).astype(np.float32)
