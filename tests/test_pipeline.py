import numpy as np

from surgiguard.models.base import CallableSegmenter
from surgiguard.pipeline import SurgiGuardPipeline


def test_pipeline_returns_controlled_mask_metrics_and_alert_state() -> None:
    segmenter = CallableSegmenter(
        lambda frame: np.full(frame.shape[:2], frame.mean() / 255.0, dtype=np.float32)
    )
    pipeline = SurgiGuardPipeline(segmenter=segmenter, threshold=0.5)
    dark = np.zeros((4, 4, 3), dtype=np.uint8)
    bright = np.full((4, 4, 3), 255, dtype=np.uint8)

    first = pipeline.process(dark, 0.0)
    second = pipeline.process(bright, 1.0)

    assert first.stabilized_mask.shape == (4, 4)
    assert first.aligned_reference.shape == (4, 4)
    assert first.change_evidence.shape == (4, 4)
    assert second.raw_probability.mean() == 1.0
    assert second.flicker == 1.0
    assert second.alert.triggered is False
