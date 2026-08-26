"""JSON schemas exposed by the public controller API."""

from pydantic import BaseModel, Field


class FrameRequest(BaseModel):
    frame: list[list[list[int]]]
    probability: list[list[float]]
    timestamp: float = Field(ge=0.0)


class FrameResponse(BaseModel):
    stabilized_probability: list[list[float]]
    gate: list[list[float]]
    timestamp: float

