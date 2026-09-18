"""FastAPI surface with isolated state for each demonstration stream."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock, RLock

import numpy as np
from fastapi import FastAPI

from ..temporal.controller import TemporalController
from .schemas import FrameRequest, FrameResponse, ResetResponse


@dataclass
class _StreamState:
    controller: TemporalController
    lock: RLock


class StreamControllerRegistry:
    def __init__(
        self,
        max_streams: int = 32,
        controller_factory: Callable[[], TemporalController] = TemporalController,
    ) -> None:
        if max_streams < 1:
            raise ValueError("max_streams must be positive")
        self.max_streams = max_streams
        self._controller_factory = controller_factory
        self._streams: dict[str, _StreamState] = {}
        self._registry_lock = Lock()

    def _state(self, stream_id: str) -> _StreamState:
        with self._registry_lock:
            existing = self._streams.get(stream_id)
            if existing is not None:
                return existing
            if len(self._streams) >= self.max_streams:
                raise RuntimeError("stream capacity reached; reset an inactive stream")
            state = _StreamState(self._controller_factory(), RLock())
            self._streams[stream_id] = state
            return state

    def process(self, stream_id: str, frame, probability, timestamp: float):
        state = self._state(stream_id)
        with state.lock:
            return state.controller.update(
                np.asarray(frame, dtype=np.uint8),
                np.asarray(probability, dtype=np.float32),
                timestamp,
            )

    def reset(self, stream_id: str) -> bool:
        with self._registry_lock:
            return self._streams.pop(stream_id, None) is not None


def create_app(registry: StreamControllerRegistry | None = None) -> FastAPI:
    streams = registry or StreamControllerRegistry()
    application = FastAPI(title="SurgiGuard Controller API", version="0.1.0")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @application.post("/v1/frames", response_model=FrameResponse)
    def process_frame(request: FrameRequest) -> FrameResponse:
        result = streams.process(
            request.stream_id,
            request.frame,
            request.probability,
            request.timestamp,
        )
        return FrameResponse(
            stabilized_probability=result.stabilized.tolist(),
            gate=result.gate.tolist(),
            timestamp=result.timestamp,
        )

    @application.post("/v1/streams/{stream_id}/reset", response_model=ResetResponse)
    def reset_stream(stream_id: str) -> ResetResponse:
        return ResetResponse(stream_id=stream_id, reset=streams.reset(stream_id))

    return application


app = create_app()
