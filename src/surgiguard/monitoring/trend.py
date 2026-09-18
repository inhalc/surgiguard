"""Short-window evolution estimates for stabilized masks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrendEstimate:
    slope: float
    current_area: float
    predicted_area: float
    sample_count: int


class AreaTrend:
    """Estimate stabilized foreground-area change with rolling least squares."""

    def __init__(self, window_size: int = 8, horizon_seconds: float = 1.0) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least two")
        self._samples: deque[tuple[float, float]] = deque(maxlen=window_size)
        self.horizon_seconds = horizon_seconds

    def update(self, timestamp: float, mask: NDArray[np.bool_]) -> TrendEstimate:
        area = float(np.asarray(mask, dtype=bool).sum())
        self._samples.append((timestamp, area))
        if len(self._samples) < 2:
            return TrendEstimate(0.0, area, area, len(self._samples))
        times = np.array([item[0] for item in self._samples], dtype=np.float64)
        areas = np.array([item[1] for item in self._samples], dtype=np.float64)
        times -= times.mean()
        denominator = float(np.dot(times, times))
        slope = 0.0 if denominator == 0.0 else float(np.dot(times, areas - areas.mean()) / denominator)
        predicted = max(0.0, area + self.horizon_seconds * slope)
        return TrendEstimate(slope, area, predicted, len(self._samples))
