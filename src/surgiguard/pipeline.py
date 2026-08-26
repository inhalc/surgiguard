"""End-to-end SurgiGuard orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models.base import FloatArray, Segmenter
from .monitoring.alerts import RiskAlert, RiskMonitor
from .monitoring.metrics import centroid_drift, mask_flicker
from .monitoring.trend import AreaTrend, TrendEstimate
from .temporal.controller import TemporalController


@dataclass(frozen=True)
class PipelineResult:
    raw_probability: FloatArray
    stabilized_probability: FloatArray
    raw_mask: NDArray[np.bool_]
    stabilized_mask: NDArray[np.bool_]
    gate: FloatArray
    flicker: float
    drift: float
    trend: TrendEstimate
    alert: RiskAlert
    timestamp: float


class SurgiGuardPipeline:
    def __init__(
        self,
        segmenter: Segmenter,
        controller: TemporalController | None = None,
        trend: AreaTrend | None = None,
        risk_monitor: RiskMonitor | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.segmenter = segmenter
        self.controller = controller or TemporalController()
        self.trend = trend or AreaTrend()
        self.risk_monitor = risk_monitor or RiskMonitor()
        self.threshold = threshold
        self._previous_mask: NDArray[np.bool_] | None = None

    def process(self, frame: NDArray[np.uint8], timestamp: float) -> PipelineResult:
        prediction = self.segmenter.predict(frame, timestamp)
        controlled = self.controller.update(frame, prediction.probability, timestamp)
        raw_mask = controlled.raw >= self.threshold
        stable_mask = controlled.stabilized >= self.threshold
        flicker = 0.0 if self._previous_mask is None else mask_flicker(self._previous_mask, stable_mask)
        drift = 0.0 if self._previous_mask is None else centroid_drift(self._previous_mask, stable_mask)
        trend = self.trend.update(timestamp, stable_mask)
        score = max(0.0, trend.predicted_area - trend.current_area) / stable_mask.size
        alert = self.risk_monitor.evaluate(score, timestamp)
        self._previous_mask = stable_mask.copy()
        return PipelineResult(
            raw_probability=controlled.raw,
            stabilized_probability=controlled.stabilized,
            raw_mask=raw_mask,
            stabilized_mask=stable_mask,
            gate=controlled.gate,
            flicker=flicker,
            drift=drift,
            trend=trend,
            alert=alert,
            timestamp=timestamp,
        )

