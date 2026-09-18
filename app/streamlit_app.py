"""SurgiGuard monitoring dashboard with a labeled synthetic stream."""

from __future__ import annotations

from pathlib import Path

from surgiguard.demo import synthetic_sequence
from surgiguard.models.base import CallableSegmenter
from surgiguard.pipeline import SurgiGuardPipeline


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
    sequence = synthetic_sequence(10)
    current_probability = sequence[step].probability
    pipeline = SurgiGuardPipeline(CallableSegmenter(lambda _: current_probability))
    result = None
    for index, item in enumerate(sequence[: step + 1]):
        pipeline.segmenter = CallableSegmenter(lambda _, value=item.probability: value)
        result = pipeline.process(item.frame, float(index))
    assert result is not None
    input_col, raw_col, reference_col, output_col = st.columns(4)
    input_col.image(sequence[step].frame, caption="Synthetic input", use_container_width=True)
    raw_col.image(
        result.raw_probability, caption="Base prediction", clamp=True, use_container_width=True
    )
    reference_col.image(
        result.aligned_reference,
        caption="Aligned history reference",
        clamp=True,
        use_container_width=True,
    )
    output_col.image(
        result.stabilized_probability,
        caption="Controlled output",
        clamp=True,
        use_container_width=True,
    )
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Flicker", f"{result.flicker:.3f}")
    metric_b.metric("Centroid drift", f"{result.drift:.2f} px")
    metric_c.metric("Gate mean", f"{result.gate.mean():.2f}")
    st.caption(f"Phase: {sequence[step].phase} · synthetic demonstration · no clinical data")


if __name__ == "__main__":
    main()
