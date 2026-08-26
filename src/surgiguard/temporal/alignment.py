"""Alignment interfaces for short-window prediction association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class AlignmentResult:
    warped: FloatArray
    confidence: FloatArray


class MaskAligner(Protocol):
    def align(self, previous: FloatArray, current_frame: NDArray[np.uint8]) -> AlignmentResult:
        """Align a previous probability map into the current frame."""


class IdentityAligner:
    """No-motion aligner for tests and already-registered streams."""

    def align(self, previous: FloatArray, current_frame: NDArray[np.uint8]) -> AlignmentResult:
        del current_frame
        mask = np.asarray(previous, dtype=np.float32)
        return AlignmentResult(mask.copy(), np.ones_like(mask, dtype=np.float32))


class FlowAligner:
    """Adapter for an externally supplied forward/backward flow implementation.

    The callable is expected to return a warped mask and a forward-backward
    consistency confidence map, both in the current image coordinates.
    """

    def __init__(self, align_fn):
        self._align_fn = align_fn

    def align(self, previous: FloatArray, current_frame: NDArray[np.uint8]) -> AlignmentResult:
        warped, confidence = self._align_fn(previous, current_frame)
        return AlignmentResult(
            np.asarray(warped, dtype=np.float32),
            np.asarray(confidence, dtype=np.float32),
        )

