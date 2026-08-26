"""Generate recruiter-facing SurgiGuard visuals from synthetic, non-clinical data."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
PAPER = "#f8f5f2"
INK = "#25201e"
MUTED = "#726862"
ACCENT = "#c74d39"
TEAL = "#2a7c78"
RULE = "#d8cec7"


def synthetic_sequence(step: int, size: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    radius = 25 + step * 2.2
    truth = ((xx - 90) ** 2 + (yy - 92) ** 2) <= radius**2
    frame = 0.2 + 0.55 * np.exp(-((xx - 90) ** 2 + (yy - 92) ** 2) / (2 * (radius + 10) ** 2))
    raw = truth.astype(float) * 0.82
    raw += 0.08 * np.sin(xx / 8 + step) * truth
    if step in (3, 4):
        frame[:, 72:110] += 0.42
        raw[:, 72:110] *= 0.15
    stable = truth.astype(float) * 0.87
    return np.clip(frame, 0, 1), np.clip(raw, 0, 1), stable


def save_hero() -> None:
    frame, raw, stable = synthetic_sequence(4)
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    grid = fig.add_gridspec(3, 12, height_ratios=[1.1, 5, 1.4], hspace=0.28, wspace=0.25)
    title = fig.add_subplot(grid[0, :])
    title.axis("off")
    title.text(0, 0.72, "SURGIGUARD / CONTROL ROOM", color=MUTED, fontsize=12, weight="bold")
    title.text(0, 0.04, "Stable when it should be. Responsive when it must be.", color=INK, fontsize=28, weight="bold")
    for index, (image, label, cmap) in enumerate(
        ((frame, "INTRAOPERATIVE STREAM", "gray"), (raw, "BASE PREDICTION", "magma"), (stable, "CONTROLLED OUTPUT", "magma"))
    ):
        ax = fig.add_subplot(grid[1, index * 4 : (index + 1) * 4])
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(label, loc="left", fontsize=10, color=MUTED, pad=10, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(RULE)
    metrics = (("FLICKER", "0.09", "publication"), ("DRIFT", "2.6 px", "publication"), ("LEAD TIME", "2.2 s", "publication"))
    for index, (name, value, note) in enumerate(metrics):
        ax = fig.add_subplot(grid[2, index * 4 : (index + 1) * 4])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.025", facecolor="#fffdfb", edgecolor=RULE))
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
        (11.8, 2.0, 2.7, 1.3, "OUTPUT", "mask + metrics + cue"),
    ]
    for x, y, width, height, title, subtitle in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#fffdfb", edgecolor=RULE, linewidth=1.4))
        ax.text(x + 0.15, y + 0.78, title, fontsize=10, color=INK, weight="bold")
        ax.text(x + 0.15, y + 0.35, subtitle, fontsize=9, color=MUTED)
    arrows = [((2.5, 2.65), (3.0, 2.65)), ((5.2, 2.65), (5.7, 3.85)), ((5.2, 2.65), (5.7, 1.4)), ((8.2, 3.85), (8.9, 2.85)), ((8.2, 1.4), (8.9, 2.45)), ((11.2, 2.65), (11.8, 2.65))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.7})
    ax.text(0.3, 4.95, "MODEL-AGNOSTIC RELIABILITY LAYER", fontsize=12, color=TEAL, weight="bold")
    fig.savefig(OUT / "architecture.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def save_sequence() -> None:
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.4), facecolor=PAPER)
    for step in range(6):
        frame, raw, stable = synthetic_sequence(step)
        for row, image in enumerate((frame, raw, stable)):
            axes[row, step].imshow(image, cmap="gray" if row == 0 else "magma", vmin=0, vmax=1)
            axes[row, step].set_xticks([])
            axes[row, step].set_yticks([])
            for spine in axes[row, step].spines.values():
                spine.set_visible(False)
    for row, label in enumerate(("Stream", "Base", "SurgiGuard")):
        axes[row, 0].set_ylabel(label, fontsize=11, color=INK, weight="bold")
    fig.suptitle("Artifact burst at frames 4–5 · synthetic illustration", x=0.08, ha="left", fontsize=14, color=INK, weight="bold")
    fig.savefig(OUT / "qualitative_results.png", dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    save_hero()
    save_architecture()
    save_sequence()
