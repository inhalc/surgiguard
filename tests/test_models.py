import numpy as np
import pytest

from surgiguard.models.base import CallableSegmenter, FramePrediction


def test_prediction_rejects_values_outside_probability_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FramePrediction(np.array([[1.2]], dtype=np.float32), timestamp=1.0)


def test_callable_segmenter_preserves_timestamp() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    segmenter = CallableSegmenter(lambda image: np.full(image.shape[:2], 0.75, dtype=np.float32))

    prediction = segmenter.predict(frame, timestamp=2.5)

    assert prediction.timestamp == 2.5
    assert prediction.probability.shape == (4, 4)
    np.testing.assert_allclose(prediction.probability, 0.75)
