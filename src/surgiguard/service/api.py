"""FastAPI surface for the model-independent temporal controller."""

import numpy as np
from fastapi import FastAPI

from ..temporal.controller import TemporalController
from .schemas import FrameRequest, FrameResponse


def create_app(controller: TemporalController | None = None) -> FastAPI:
    temporal = controller or TemporalController()
    application = FastAPI(title="SurgiGuard Controller API", version="0.1.0")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @application.post("/v1/frames", response_model=FrameResponse)
    def process_frame(request: FrameRequest) -> FrameResponse:
        result = temporal.update(
            np.asarray(request.frame, dtype=np.uint8),
            np.asarray(request.probability, dtype=np.float32),
            request.timestamp,
        )
        return FrameResponse(
            stabilized_probability=result.stabilized.tolist(),
            gate=result.gate.tolist(),
            timestamp=result.timestamp,
        )

    return application


app = create_app()

