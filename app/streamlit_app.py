"""SurgiGuard monitoring dashboard with a labeled synthetic stream."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from surgiguard.models.base import CallableSegmenter
from surgiguard.pipeline import SurgiGuardPipeline


def synthetic_frame(step: int, size: int = 192) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    radius = 28 + step * 2
    probability = np.exp(-((xx - 96) ** 2 + (yy - 96) ** 2) / (2.0 * radius**2)).astype(np.float32)
    frame = np.repeat((probability * 185 + 30).astype(np.uint8)[..., None], 3, axis=2)
    if step == 5:
        frame[:, 75:110] = 245
        probability[:, 75:110] *= 0.15
    return frame, probability


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SurgiGuard", page_icon="SG", layout="wide")
    root = Path(__file__).resolve().parents[1]
    css = (root / "tokens.css").read_text(encoding="utf-8")
    css += (root / "app" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.title("SurgiGuard Control Room")
    st.caption("Synthetic sequence · no clinical data is included in this public demo")
    step = st.slider("Sequence frame", 0, 9, 4)
    predictions = {index: synthetic_frame(index) for index in range(step + 1)}
    current_probability = predictions[step][1]
    pipeline = SurgiGuardPipeline(CallableSegmenter(lambda _: current_probability))
    result = None
    for index, (frame, probability) in predictions.items():
        pipeline.segmenter = CallableSegmenter(lambda _, value=probability: value)
        result = pipeline.process(frame, float(index))
    assert result is not None
    left, middle, right = st.columns(3)
    left.image(predictions[step][0], caption="Input stream", use_container_width=True)
    middle.image(result.raw_probability, caption="Base prediction", clamp=True, use_container_width=True)
    right.image(result.stabilized_probability, caption="Controlled output", clamp=True, use_container_width=True)
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Flicker", f"{result.flicker:.3f}")
    metric_b.metric("Centroid drift", f"{result.drift:.2f} px")
    metric_c.metric("Gate mean", f"{result.gate.mean():.2f}")



if __name__ == "__main__":
    main()
