"""Run SurgiGuard with a user-supplied segmentation plugin."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import cv2

from surgiguard.models.base import CallableSegmenter
from surgiguard.pipeline import SurgiGuardPipeline


def load_callable(specification: str):
    module_name, function_name = specification.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--predictor", required=True, help="Python path in module:function form")
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    pipeline = SurgiGuardPipeline(CallableSegmenter(load_callable(args.predictor)))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = pipeline.process(frame, index / fps)
        print(f"frame={index} flicker={result.flicker:.4f} drift={result.drift:.2f}")
        index += 1


if __name__ == "__main__":
    main()
