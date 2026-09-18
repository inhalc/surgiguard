"""Model-agnostic segmentation interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class FramePrediction:
    """A per-frame foreground probability map."""

    probability: FloatArray
    timestamp: float

    def __post_init__(self) -> None:
        probability = np.asarray(self.probability, dtype=np.float32)
        if probability.ndim != 2:
            raise ValueError("probability must be a 2D array")
        if not np.isfinite(probability).all():
            raise ValueError("probability must contain finite values")
        if probability.min(initial=0.0) < 0.0 or probability.max(initial=1.0) > 1.0:
            raise ValueError("probability values must lie in [0, 1]")
        object.__setattr__(self, "probability", probability)


class Segmenter(ABC):
    """Interface implemented by any per-frame segmentation model."""

    @abstractmethod
    def predict(self, frame: ImageArray, timestamp: float) -> FramePrediction:
        """Return a foreground probability map for one frame."""


class CallableSegmenter(Segmenter):
    """Adapter for a callable that maps an image to a probability map."""

    def __init__(self, predictor: Callable[[ImageArray], FloatArray]) -> None:
        self._predictor = predictor

    def predict(self, frame: ImageArray, timestamp: float) -> FramePrediction:
        return FramePrediction(self._predictor(frame), timestamp)
