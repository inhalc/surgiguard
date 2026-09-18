"""Generate SurgiGuard pipeline diagrams and synthetic demonstration figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surgiguard.demo import synthetic_sequence
from surgiguard.models.base import CallableSegmenter
from surgiguard.pipeline import SurgiGuardPipeline

OUT = Path(__file__).resolve().parents[1]
PAPER = "#f8f5f2"
INK = "#25201e"
MUTED = "#726862"
ACCENT = "#c74d39"
TEAL = "#2a7c78"
RULE = "#d8cec7"


def run_demo(length: int = 9, size: int = 180):
    sequence = synthetic_sequence(length=length, size=size)
    pipeline = SurgiGuardPipeline(CallableSegmenter(lambda _: sequence[0].probability))
    results = []
    for index, item in enumerate(sequence):
        pipeline.segmenter = CallableSegmenter(lambda _, value=item.probability: value)
        results.append(pipeline.process(item.frame, float(index)))
    return sequence, results


def save_hero() -> None:
    sequence, results = run_demo()
    item, result = sequence[4], results[4]
    fig = plt.figure(figsize=(14, 6.1), facecolor=PAPER)
    fig.text(0.045, 0.925, "SURGIGUARD  /  TEMPORAL INFERENCE", color=TEAL,
             fontsize=11, weight="bold")
    fig.text(0.045, 0.845, "From frame predictions to a controlled video stream",
             color=INK, fontsize=22, weight="bold")
    fig.text(0.045, 0.79, "Fixed segmenter  /  Motion-aligned history  /  Per-pixel update gate",
             color=MUTED, fontsize=11)
    panels = [
        (item.frame, "01  INPUT FRAME", "Synthetic acquisition", None),
        (result.raw_probability, "02  BASE PREDICTION", "Current model probability", "magma"),
        (result.aligned_reference, "03  ALIGNED HISTORY", "Reference in target coordinates", "magma"),
        (result.stabilized_probability, "04  CONTROLLED OUTPUT", "History + current evidence", "magma"),
    ]
    for i, (values, title, subtitle, cmap) in enumerate(panels):
        x = 0.045 + i * 0.238
        ax = fig.add_axes([x, 0.235, 0.205, 0.47])
        if cmap is None:
            ax.imshow(values)
        else:
            ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
        ax.set_axis_off()
        fig.text(x, 0.735, title, color=TEAL if i == 3 else INK,
                 fontsize=10, weight="bold")
        fig.text(x, 0.195, subtitle, color=MUTED, fontsize=9)
    fig.text(0.045, 0.105, "INSPECT THE UPDATE", color=TEAL, fontsize=10, weight="bold")
    fig.text(0.25, 0.105, "Motion  →  transient artifact  →  recovery  →  area growth",
             color=INK, fontsize=11)
    fig.text(0.045, 0.04, "Synthetic demonstration · Actual pipeline output · Probability maps use a shared 0–1 scale",
             color=MUTED, fontsize=9)
    fig.savefig(OUT / "hero.png", dpi=220, facecolor=PAPER)
    plt.close(fig)


def save_architecture() -> None:
    fig, ax = plt.subplots(figsize=(15, 5.4), facecolor=PAPER)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    boxes = [
        (0.3, 2.0, 2.2, 1.3, "VIDEO STREAM", "frame I_t"),
        (3.0, 2.0, 2.2, 1.3, "BASE MODEL", "probability P_t"),
        (5.7, 3.25, 2.5, 1.3, "HISTORY", "align + aggregate"),
        (5.7, 0.75, 2.5, 1.3, "CHANGE EVIDENCE", "appearance delta I_t"),
        (8.9, 2.0, 2.3, 1.3, "STABILITY GATE", "soft update control"),
        (11.8, 2.0, 2.7, 1.3, "OUTPUT", "mask + metrics + trend"),
    ]
    for x, y, width, height, title, subtitle in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#fffdfb",
                edgecolor=RULE,
                linewidth=1.4,
            )
        )
        ax.text(x + 0.15, y + 0.78, title, fontsize=10, color=INK, weight="bold")
        ax.text(x + 0.15, y + 0.35, subtitle, fontsize=9, color=MUTED)
    arrows = [
        ((2.5, 2.65), (3.0, 2.65)),
        ((5.2, 2.65), (5.7, 3.85)),
        ((5.2, 2.65), (5.7, 1.4)),
        ((8.2, 3.85), (8.9, 2.85)),
        ((8.2, 1.4), (8.9, 2.45)),
        ((11.2, 2.65), (11.8, 2.65)),
    ]
    for start, end in arrows:
        ax.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.7}
        )
    ax.text(0.3, 4.95, "MODEL-AGNOSTIC RELIABILITY LAYER", fontsize=12, color=TEAL, weight="bold")
    fig.savefig(OUT / "architecture.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def save_sequence() -> None:
    sequence, results = run_demo(length=9, size=128)
    fig, axes = plt.subplots(4, 9, figsize=(17, 8.2), facecolor=PAPER)
    for step, (item, result) in enumerate(zip(sequence, results)):
        images = (
            item.frame,
            result.raw_probability,
            result.aligned_reference,
            result.stabilized_probability,
        )
        for row, image in enumerate(images):
            if row == 0:
                axes[row, step].imshow(image)
            else:
                axes[row, step].imshow(image, cmap="magma", vmin=0, vmax=1)
            axes[row, step].set_xticks([])
            axes[row, step].set_yticks([])
            for spine in axes[row, step].spines.values():
                spine.set_visible(False)
        axes[0, step].set_title(item.phase, fontsize=8, color=MUTED)
    for row, label in enumerate(("Input", "Base", "Aligned history", "Controlled")):
        axes[row, 0].set_ylabel(label, fontsize=11, color=INK, weight="bold")
    fig.suptitle(
        "Motion → transient artifact → true growth · synthetic pipeline run",
        x=0.08,
        ha="left",
        fontsize=14,
        color=INK,
        weight="bold",
    )
    fig.savefig(OUT / "qualitative_results.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    save_hero()
    save_architecture()
    save_sequence()
