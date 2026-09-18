"""Reliability-filtered history reference estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _sigmoid(value: FloatArray) -> FloatArray:
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def robust_reference(
    aligned_masks: FloatArray,
    confidences: FloatArray,
    min_confidence: float,
    trim_fraction: float,
) -> tuple[FloatArray, NDArray[np.integer]]:
    """Compute the paper's per-pixel trimmed, confidence-weighted reference."""

    masks = np.asarray(aligned_masks, dtype=np.float32)
    weights = np.asarray(confidences, dtype=np.float32)
    if masks.ndim != 3 or masks.shape != weights.shape:
        raise ValueError("aligned_masks and confidences must share shape [history, height, width]")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")

    reliable = weights >= min_confidence
    reliable_count = reliable.sum(axis=0)
    height, width = masks.shape[1:]
    reference = np.zeros((height, width), dtype=np.float32)

    for row in range(height):
        for col in range(width):
            indices = np.flatnonzero(reliable[:, row, col])
            if indices.size == 0:
                continue
            ordered = indices[np.argsort(masks[indices, row, col])]
            trim = min(int(np.floor(trim_fraction * ordered.size)), (ordered.size - 1) // 2)
            retained = ordered[trim : ordered.size - trim] if trim else ordered
            retained_weights = weights[retained, row, col]
            numerator = np.sum(retained_weights * masks[retained, row, col])
            reference[row, col] = numerator / (retained_weights.sum() + 1e-8)

    return reference, reliable_count


def mix_unreliable_reference(
    reference: FloatArray,
    current: FloatArray,
    reliable_count: NDArray[np.integer],
    required_count: float,
    count_temperature: float,
) -> FloatArray:
    """Fall back smoothly to the current prediction when history is sparse."""

    weight = _sigmoid((reliable_count.astype(np.float32) - required_count) / count_temperature)
    return weight * reference + (1.0 - weight) * current
