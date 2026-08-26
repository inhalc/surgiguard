import numpy as np

from surgiguard.temporal.alignment import IdentityAligner
from surgiguard.temporal.controller import TemporalController
from surgiguard.temporal.history_reference import robust_reference
from surgiguard.temporal.stability_gate import compute_stability_gate


def test_identity_alignment_preserves_mask() -> None:
    mask = np.array([[0.1, 0.8], [0.2, 0.9]], dtype=np.float32)
    result = IdentityAligner().align(mask, np.zeros((2, 2, 3), dtype=np.uint8))
    np.testing.assert_allclose(result.warped, mask)
    np.testing.assert_allclose(result.confidence, 1.0)


def test_robust_reference_rejects_single_outlier() -> None:
    masks = np.stack([
        np.full((2, 2), 0.2, dtype=np.float32),
        np.full((2, 2), 0.22, dtype=np.float32),
        np.full((2, 2), 0.95, dtype=np.float32),
    ])
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
