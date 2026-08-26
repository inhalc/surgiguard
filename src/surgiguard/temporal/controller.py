"""Stateful stability-gated temporal inference controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .alignment import IdentityAligner, MaskAligner
from .history_reference import mix_unreliable_reference, robust_reference
from .stability_gate import compute_stability_gate, image_change_evidence

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class ControlledPrediction:
    raw: FloatArray
    reference: FloatArray
    gate: FloatArray
    stabilized: FloatArray
    change_evidence: FloatArray
    timestamp: float


class TemporalController:
    """Maintain a short reliable history and regulate each frame update."""

    def __init__(
        self,
        history_size: int = 5,
        aligner: MaskAligner | None = None,
        min_confidence: float = 0.5,
        trim_fraction: float = 0.2,
        required_count: float = 2.0,
        count_temperature: float = 0.75,
        disagreement_threshold: float = 0.2,
        change_threshold: float = 0.2,
        disagreement_temperature: float = 0.05,
        change_temperature: float = 0.05,
        change_low: float = 0.04,
        change_high: float = 0.35,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if change_high <= change_low:
            raise ValueError("change_high must exceed change_low")
        self._history: deque[FloatArray] = deque(maxlen=history_size)
        self._aligner = aligner or IdentityAligner()
        self._previous_frame: NDArray[np.uint8] | None = None
        self.min_confidence = min_confidence
        self.trim_fraction = trim_fraction
        self.required_count = required_count
        self.count_temperature = count_temperature
        self.disagreement_threshold = disagreement_threshold
        self.change_threshold = change_threshold
        self.disagreement_temperature = disagreement_temperature
        self.change_temperature = change_temperature
        self.change_low = change_low
        self.change_high = change_high

    def update(
        self,
        frame: NDArray[np.uint8],
        probability: FloatArray,
        timestamp: float,
    ) -> ControlledPrediction:
        current = np.asarray(probability, dtype=np.float32)
        if not self._history:
            zeros = np.zeros_like(current)
            result = ControlledPrediction(current, current, zeros, current, zeros, timestamp)
        else:
            aligned = [self._aligner.align(mask, frame) for mask in self._history]
            masks = np.stack([item.warped for item in aligned])
            confidence = np.stack([item.confidence for item in aligned])
            raw_reference, count = robust_reference(
                masks, confidence, self.min_confidence, self.trim_fraction
            )
            reference = mix_unreliable_reference(
                raw_reference,
                current,
                count,
                self.required_count,
                self.count_temperature,
            )
            change = image_change_evidence(
                self._previous_frame, frame, self.change_low, self.change_high
            )
            gate = compute_stability_gate(
                current - reference,
                change,
                self.disagreement_threshold,
                self.change_threshold,
                self.disagreement_temperature,
                self.change_temperature,
            )
            stabilized = gate * reference + (1.0 - gate) * current
            result = ControlledPrediction(current, reference, gate, stabilized, change, timestamp)

        self._history.append(result.stabilized.copy())
        self._previous_frame = np.asarray(frame, dtype=np.uint8).copy()
        return result

