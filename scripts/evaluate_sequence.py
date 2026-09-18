"""Evaluate stored binary masks with SurgiGuard reliability metrics."""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import numpy as np

from surgiguard.monitoring.metrics import centroid_drift, mask_flicker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("masks", type=Path, help="NPZ file containing [time, height, width] masks")
    args = parser.parse_args()
    masks = np.load(args.masks)["masks"].astype(bool)
    flicker = [mask_flicker(a, b) for a, b in pairwise(masks)]
    drift = [centroid_drift(a, b) for a, b in pairwise(masks)]
    print(f"mean_flicker={np.mean(flicker):.4f}")
    print(f"mean_drift={np.mean(drift):.2f}px")


if __name__ == "__main__":
    main()
