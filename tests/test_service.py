import numpy as np
import pytest

from surgiguard.service.api import StreamControllerRegistry


def frame(value: int) -> list[list[list[int]]]:
    return np.full((2, 2, 3), value, dtype=np.uint8).tolist()


def probability(value: float) -> list[list[float]]:
    return np.full((2, 2), value, dtype=np.float32).tolist()


def test_registry_isolates_interleaved_stream_state() -> None:
    registry = StreamControllerRegistry(max_streams=2)
    registry.process("a", frame(0), probability(0.2), 0.0)
    registry.process("b", frame(255), probability(0.8), 0.0)
    interleaved_a = registry.process("a", frame(0), probability(0.4), 1.0)

    independent = StreamControllerRegistry(max_streams=1)
    independent.process("a", frame(0), probability(0.2), 0.0)
    independent_a = independent.process("a", frame(0), probability(0.4), 1.0)
    np.testing.assert_allclose(interleaved_a.stabilized, independent_a.stabilized)


def test_registry_reset_restores_first_frame_behavior() -> None:
    registry = StreamControllerRegistry(max_streams=1)
    registry.process("a", frame(0), probability(0.2), 0.0)
    registry.reset("a")
    result = registry.process("a", frame(0), probability(0.8), 0.0)
    np.testing.assert_allclose(result.stabilized, 0.8)


def test_registry_rejects_capacity_overflow() -> None:
    registry = StreamControllerRegistry(max_streams=1)
    registry.process("a", frame(0), probability(0.2), 0.0)
    with pytest.raises(RuntimeError, match="capacity"):
        registry.process("b", frame(0), probability(0.2), 0.0)
