"""Alignment interfaces for short-window prediction association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class AlignmentResult:
    warped: FloatArray
    confidence: FloatArray


class MaskAligner(Protocol):
    def align(
        self,
        source_frame: NDArray[np.uint8],
        target_frame: NDArray[np.uint8],
        source_probability: FloatArray,
    ) -> AlignmentResult:
        """Align a previous probability map into the current frame."""


class IdentityAligner:
    """No-motion aligner for tests and already-registered streams."""

    def align(
        self,
        source_frame: NDArray[np.uint8],
        target_frame: NDArray[np.uint8],
        source_probability: FloatArray,
    ) -> AlignmentResult:
        del source_frame, target_frame
        mask = np.asarray(source_probability, dtype=np.float32)
        return AlignmentResult(mask.copy(), np.ones_like(mask, dtype=np.float32))


def warp_with_backward_flow(
    source_probability: FloatArray,
    backward_flow: FloatArray,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Sample a source probability map at target-to-source flow coordinates."""

    source = np.asarray(source_probability, dtype=np.float32)
    flow = np.asarray(backward_flow, dtype=np.float32)
    if flow.shape != (*source.shape, 2):
        raise ValueError("backward_flow must have shape [height, width, 2]")
    height, width = source.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    warped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return warped.astype(np.float32), valid


def _gray(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    raise ValueError("frame must have shape [height, width] or [height, width, 3]")


class DenseFlowAligner:
    """OpenCV dense-flow alignment for the public demonstration."""

    def __init__(self, consistency_sigma: float = 1.5) -> None:
        if consistency_sigma <= 0:
            raise ValueError("consistency_sigma must be positive")
        self.consistency_sigma = consistency_sigma

    @staticmethod
    def _flow(source: NDArray[np.uint8], target: NDArray[np.uint8]) -> FloatArray:
        return cv2.calcOpticalFlowFarneback(
            source,
            target,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        ).astype(np.float32)

    def align(
        self,
        source_frame: NDArray[np.uint8],
        target_frame: NDArray[np.uint8],
        source_probability: FloatArray,
    ) -> AlignmentResult:
        source_gray = _gray(source_frame)
        target_gray = _gray(target_frame)
        probability = np.asarray(source_probability, dtype=np.float32)
        if source_gray.shape != target_gray.shape or probability.shape != source_gray.shape:
            raise ValueError("source frame, target frame, and probability must share spatial shape")

        forward = self._flow(source_gray, target_gray)
        backward = self._flow(target_gray, source_gray)
        warped, valid = warp_with_backward_flow(probability, backward)
        sampled_forward_x, _ = warp_with_backward_flow(forward[..., 0], backward)
        sampled_forward_y, _ = warp_with_backward_flow(forward[..., 1], backward)
        error = np.sqrt(
            (sampled_forward_x + backward[..., 0]) ** 2
            + (sampled_forward_y + backward[..., 1]) ** 2
        )
        confidence = np.exp(-0.5 * (error / self.consistency_sigma) ** 2)
        confidence = (confidence * valid).astype(np.float32)
        return AlignmentResult(warped, confidence)


class FlowAligner:
    """Adapter for an externally supplied forward/backward alignment implementation.

    The callable is expected to return a warped mask and a forward-backward
    consistency confidence map, both in the current image coordinates.
    """

    def __init__(self, align_fn):
        self._align_fn = align_fn

    def align(
        self,
        source_frame: NDArray[np.uint8],
        target_frame: NDArray[np.uint8],
        source_probability: FloatArray,
    ) -> AlignmentResult:
        warped, confidence = self._align_fn(source_frame, target_frame, source_probability)
        return AlignmentResult(
            np.asarray(warped, dtype=np.float32),
            np.asarray(confidence, dtype=np.float32),
        )
