import numpy as np

from surgiguard.monitoring.alerts import RiskMonitor
from surgiguard.monitoring.metrics import centroid_drift, mask_flicker
from surgiguard.monitoring.trend import AreaTrend


def test_reliability_metrics_are_zero_for_unchanged_mask() -> None:
    mask = np.array([[0, 1], [0, 1]], dtype=bool)
    assert mask_flicker(mask, mask) == 0.0
    assert centroid_drift(mask, mask) == 0.0


def test_centroid_drift_measures_translation() -> None:
    previous = np.zeros((5, 5), dtype=bool)
    current = np.zeros((5, 5), dtype=bool)
    previous[2, 1] = True
    current[2, 4] = True
    assert centroid_drift(previous, current) == 3.0


def test_area_trend_detects_monotonic_growth() -> None:
    trend = AreaTrend(window_size=4)
    estimate = None
    for timestamp, area in enumerate((1, 2, 3, 4)):
        mask = np.zeros((2, 2), dtype=bool)
        mask.flat[:area] = True
        estimate = trend.update(float(timestamp), mask)
    assert estimate is not None
    assert estimate.slope > 0.9


def test_risk_monitor_requires_persistent_score_and_respects_cooldown() -> None:
    monitor = RiskMonitor(score_threshold=0.1, persistence=2, cooldown_seconds=5.0)
    assert not monitor.evaluate(score=0.2, timestamp=0.0).triggered
    assert monitor.evaluate(score=0.2, timestamp=1.0).triggered
    assert not monitor.evaluate(score=0.3, timestamp=2.0).triggered
    assert monitor.evaluate(score=0.3, timestamp=2.0).reason == "monitoring area-growth trend"
