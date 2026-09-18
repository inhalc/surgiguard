"""Deterministic synthetic sequence used by the public demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DemoFrame:
    frame: NDArray[np.uint8]
    probability: NDArray[np.float32]
    truth: NDArray[np.bool_]
    phase: str


def synthetic_sequence(length: int = 9, size: int = 192) -> tuple[DemoFrame, ...]:
    if length < 1 or size < 32:
        raise ValueError("length must be positive and size must be at least 32")
    yy, xx = np.mgrid[:size, :size]
    sequence: list[DemoFrame] = []
    for step in range(length):
        center_x = int(size * 0.38) + min(step, 2) * max(2, size // 32)
        center_y = int(size * 0.5)
        radius = int(size * 0.15) + max(0, step - 5) * max(1, size // 64)
        truth = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        texture = 18.0 * np.sin(xx / 7.0) + 12.0 * np.cos(yy / 9.0)
        luminance = 45.0 + texture + truth.astype(np.float32) * 145.0
        probability = truth.astype(np.float32) * 0.9
        if step in (3, 4):
            phase = "artifact"
            left = max(0, center_x - radius // 2)
            right = min(size, center_x + radius // 2)
            luminance[:, left:right] += 55.0
            probability[:, left:right] *= 0.08
        elif step < 3:
            phase = "motion"
        elif step < 6:
            phase = "recovery"
        else:
            phase = "growth"
        frame = np.repeat(np.clip(luminance, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        sequence.append(DemoFrame(frame, probability, truth, phase))
    return tuple(sequence)
