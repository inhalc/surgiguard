"""Generate recruiter-facing SurgiGuard visuals from synthetic, non-clinical data."""

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
    item = sequence[4]
    result = results[4]
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    grid = fig.add_gridspec(3, 12, height_ratios=[1.1, 5, 1.4], hspace=0.28, wspace=0.25)
    title = fig.add_subplot(grid[0, :])
    title.axis("off")
    title.text(0, 0.72, "SURGIGUARD / CONTROL ROOM", color=MUTED, fontsize=12, weight="bold")
    title.text(
        0,
        0.04,
        "Stable when it should be. Responsive when it must be.",
        color=INK,
        fontsize=28,
        weight="bold",
    )
    for index, (image, label, cmap) in enumerate(
        (
            (item.frame, "SYNTHETIC INPUT", None),
            (result.raw_probability, "BASE PREDICTION", "magma"),
            (result.aligned_reference, "ALIGNED REFERENCE", "magma"),
            (result.stabilized_probability, "CONTROLLED OUTPUT", "magma"),
        )
    ):
        ax = fig.add_subplot(grid[1, index * 3 : (index + 1) * 3])
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(label, loc="left", fontsize=10, color=MUTED, pad=10, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(RULE)
    metrics = (
        ("FLICKER", f"{result.flicker:.3f}", "synthetic"),
        ("DRIFT", f"{result.drift:.2f} px", "synthetic"),
        ("GATE MEAN", f"{result.gate.mean():.2f}", "synthetic"),
    )
    for index, (name, value, note) in enumerate(metrics):
        ax = fig.add_subplot(grid[2, index * 4 : (index + 1) * 4])
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                boxstyle="round,pad=0.01,rounding_size=0.025",
                facecolor="#fffdfb",
                edgecolor=RULE,
            )
        )
        ax.text(0.05, 0.67, name, fontsize=10, color=MUTED, weight="bold")
        ax.text(0.05, 0.22, value, fontsize=23, color=TEAL, weight="bold")
        ax.text(0.95, 0.24, note, fontsize=8, color=MUTED, ha="right")
    fig.savefig(OUT / "hero.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
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
