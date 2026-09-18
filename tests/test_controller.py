import numpy as np
import pytest

from surgiguard.temporal.alignment import (
    DenseFlowAligner,
    IdentityAligner,
    warp_with_backward_flow,
)
from surgiguard.temporal.controller import TemporalController
from surgiguard.temporal.history_reference import robust_reference
from surgiguard.temporal.stability_gate import compute_stability_gate


def test_identity_alignment_preserves_mask() -> None:
    mask = np.array([[0.1, 0.8], [0.2, 0.9]], dtype=np.float32)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = IdentityAligner().align(frame, frame, mask)
    np.testing.assert_allclose(result.warped, mask)
    np.testing.assert_allclose(result.confidence, 1.0)


def test_robust_reference_rejects_single_outlier() -> None:
    masks = np.stack(
        [
            np.full((2, 2), 0.2, dtype=np.float32),
            np.full((2, 2), 0.22, dtype=np.float32),
            np.full((2, 2), 0.95, dtype=np.float32),
        ]
    )
    confidence = np.ones_like(masks)
    reference, reliable_count = robust_reference(masks, confidence, 0.5, 0.34)
    np.testing.assert_allclose(reference, 0.22, atol=1e-5)
    np.testing.assert_array_equal(reliable_count, 3)


def test_gate_decreases_with_disagreement_or_change() -> None:
    stable = compute_stability_gate(np.array([[0.01]]), np.array([[0.01]]), 0.2, 0.2, 0.05, 0.05)
    disagree = compute_stability_gate(np.array([[0.8]]), np.array([[0.01]]), 0.2, 0.2, 0.05, 0.05)
    changing = compute_stability_gate(np.array([[0.01]]), np.array([[0.8]]), 0.2, 0.2, 0.05, 0.05)
    assert stable.item() > disagree.item()
    assert stable.item() > changing.item()


def test_controller_anchors_stable_stream_and_releases_on_change() -> None:
    controller = TemporalController(history_size=3, change_low=0.02, change_high=0.2)
    dark = np.zeros((2, 2, 3), dtype=np.uint8)
    bright = np.full((2, 2, 3), 255, dtype=np.uint8)
    controller.update(dark, np.full((2, 2), 0.2, dtype=np.float32), 0.0)
    stable = controller.update(dark, np.full((2, 2), 0.4, dtype=np.float32), 1.0)
    changed = controller.update(bright, np.full((2, 2), 0.9, dtype=np.float32), 2.0)
    assert stable.stabilized.mean() < 0.4
    assert changed.stabilized.mean() > stable.stabilized.mean()
    assert changed.gate.mean() < stable.gate.mean()


def test_backward_flow_warps_source_probability_into_target_coordinates() -> None:
    source = np.zeros((3, 5), dtype=np.float32)
    source[1, 1] = 1.0
    # For an object translated one pixel right, each target pixel samples x - 1.
    backward = np.zeros((3, 5, 2), dtype=np.float32)
    backward[..., 0] = -1.0
    warped, valid = warp_with_backward_flow(source, backward)
    assert warped[1, 2] == pytest.approx(1.0)
    assert valid[:, 0].sum() == 0
    assert valid[:, 1:].all()


def test_controller_rejects_non_finite_probability_and_non_increasing_time() -> None:
    controller = TemporalController()
    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    controller.update(frame, np.zeros((3, 3), dtype=np.float32), 1.0)
    with pytest.raises(ValueError, match="strictly increase"):
        controller.update(frame, np.zeros((3, 3), dtype=np.float32), 1.0)
    invalid = np.zeros((3, 3), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TemporalController().update(frame, invalid, 0.0)


def test_controller_passes_each_historical_source_frame_to_aligner() -> None:
    calls: list[tuple[int, int]] = []

    class RecordingAligner:
        def align(self, source_frame, target_frame, source_probability):
            calls.append((int(source_frame[0, 0, 0]), int(target_frame[0, 0, 0])))
            return IdentityAligner().align(source_frame, target_frame, source_probability)

    controller = TemporalController(history_size=3, aligner=RecordingAligner())
    probability = np.full((2, 2), 0.3, dtype=np.float32)
    for index in range(3):
        frame = np.full((2, 2, 3), index, dtype=np.uint8)
        controller.update(frame, probability, float(index))
    assert calls == [(0, 1), (0, 2), (1, 2)]


def test_controller_uses_motion_alignment_by_default() -> None:
    assert isinstance(TemporalController()._aligner, DenseFlowAligner)


def test_dense_flow_alignment_tracks_synthetic_translation() -> None:
    source_frame = np.zeros((64, 64), dtype=np.uint8)
    target_frame = np.zeros((64, 64), dtype=np.uint8)
    source_frame[20:40, 15:35] = 255
    target_frame[20:40, 19:39] = 255
    probability = (source_frame > 0).astype(np.float32)
    result = DenseFlowAligner().align(source_frame, target_frame, probability)
    source_x = np.argwhere(probability > 0.5)[:, 1].mean()
    warped_x = np.argwhere(result.warped > 0.5)[:, 1].mean()
    assert warped_x > source_x + 2.0
    assert result.confidence.mean() > 0.5
