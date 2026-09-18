"""Evaluate stored binary masks with SurgiGuard reliability metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from surgiguard.monitoring.metrics import centroid_drift, mask_flicker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("masks", type=Path, help="NPZ file containing [time, height, width] masks")
    args = parser.parse_args()
    masks = np.load(args.masks)["masks"].astype(bool)
    flicker = [mask_flicker(a, b) for a, b in zip(masks[:-1], masks[1:])]
    drift = [centroid_drift(a, b) for a, b in zip(masks[:-1], masks[1:])]
    print(f"mean_flicker={np.mean(flicker):.4f}")
    print(f"mean_drift={np.mean(drift):.2f}px")


if __name__ == "__main__":
    main()
