"""Sequence-level reliability metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mask_flicker(previous: NDArray[np.bool_], current: NDArray[np.bool_]) -> float:
    """Fraction of pixels whose binary state changes between frames."""

    before = np.asarray(previous, dtype=bool)
    after = np.asarray(current, dtype=bool)
    if before.shape != after.shape:
        raise ValueError("masks must share shape")
    return float(np.logical_xor(before, after).mean())


def centroid_drift(previous: NDArray[np.bool_], current: NDArray[np.bool_]) -> float:
    """Euclidean foreground-centroid displacement in pixels."""

    before_points = np.argwhere(np.asarray(previous, dtype=bool))
    after_points = np.argwhere(np.asarray(current, dtype=bool))
    if before_points.size == 0 and after_points.size == 0:
        return 0.0
    if before_points.size == 0 or after_points.size == 0:
        return float("inf")
    return float(np.linalg.norm(after_points.mean(axis=0) - before_points.mean(axis=0)))

