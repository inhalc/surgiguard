"""Persistent, cooldown-aware operational risk prompts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskAlert:
    triggered: bool
    score: float
    timestamp: float
    reason: str


class RiskMonitor:
    def __init__(
        self,
        score_threshold: float = 0.05,
        persistence: int = 3,
        cooldown_seconds: float = 5.0,
    ) -> None:
        if persistence < 1:
            raise ValueError("persistence must be positive")
        self.score_threshold = score_threshold
        self.persistence = persistence
        self.cooldown_seconds = cooldown_seconds
        self._consecutive = 0
        self._last_trigger = float("-inf")

    def evaluate(self, score: float, timestamp: float) -> RiskAlert:
        self._consecutive = self._consecutive + 1 if score >= self.score_threshold else 0
        cooldown_ready = timestamp - self._last_trigger >= self.cooldown_seconds
        triggered = self._consecutive >= self.persistence and cooldown_ready
        if triggered:
            self._last_trigger = timestamp
            self._consecutive = 0
        reason = (
            "persistent area-growth trend detected" if triggered else "monitoring area-growth trend"
        )
        return RiskAlert(triggered, float(score), float(timestamp), reason)
