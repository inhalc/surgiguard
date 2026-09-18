"""JSON schemas exposed by the public controller API."""

from pydantic import BaseModel, Field


class FrameRequest(BaseModel):
    stream_id: str = Field(min_length=1, max_length=128)
    frame: list[list[list[int]]]
    probability: list[list[float]]
    timestamp: float = Field(ge=0.0)


class FrameResponse(BaseModel):
    stabilized_probability: list[list[float]]
    gate: list[list[float]]
    timestamp: float


class ResetResponse(BaseModel):
    stream_id: str
    reset: bool
