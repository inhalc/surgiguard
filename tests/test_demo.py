import numpy as np

from surgiguard.demo import synthetic_sequence


def test_synthetic_sequence_contains_motion_artifact_and_true_growth() -> None:
    sequence = synthetic_sequence(9, size=96)
    assert [item.phase for item in sequence] == [
        "motion",
        "motion",
        "motion",
        "artifact",
        "artifact",
        "recovery",
        "growth",
        "growth",
        "growth",
    ]
    early_x = np.argwhere(sequence[0].truth)[:, 1].mean()
    later_x = np.argwhere(sequence[2].truth)[:, 1].mean()
    assert later_x > early_x
    assert sequence[3].probability.sum() < sequence[2].probability.sum()
    assert sequence[-1].truth.sum() > sequence[5].truth.sum()
