"""Adapter boundary for externally supplied MedSAM predictors."""

from __future__ import annotations

from collections.abc import Callable

from .base import CallableSegmenter, FloatArray, ImageArray


class MedSAMAdapter(CallableSegmenter):
    """Expose an authorized MedSAM inference callable through SurgiGuard.

    Model construction and weights intentionally remain outside this public
    repository. The callable must return a two-dimensional probability map.
    """

    def __init__(self, predictor: Callable[[ImageArray], FloatArray]) -> None:
        super().__init__(predictor)

