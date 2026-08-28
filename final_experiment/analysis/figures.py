"""Deterministic publication figures sourced only from generated results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_STEMS = (
    "pipeline_overview",
    "overall_comparison",
    "confusion_matrices",
)
COLORS = {"C1": "#0072B2", "C3": "#009E73", "C4": "#D55E00"}
PIPELINE_LABELS = {
    "C1": "ProofWriter → Base Qwen3-1.7B → Direct prediction",
    "C3": "ProofWriter → Qwen3-1.7B + LoRA checkpoint-250 → Direct prediction",
    "C4": "ProofWriter → Qwen3-1.7B + LoRA checkpoint-250 (same as C3) → NL2Logic → AST → Z3 → Symbolic prediction",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def _labels(ax: plt.Axes, bars: Any, percent: bool = True) -> None:
    for bar in bars:
        value = bar.get_height()
        text = f"{100 * value:.1f}" if percent else f"{int(value)}"
        ax.text(bar.get_x() + bar.get_width() / 2, value, text, ha="center", va="bottom", fontsize=8)


def pipeline_geometry(node_count: int) -> tuple[np.ndarray, float]:
    width = min(0.19, 0.70 / node_count)
    centers = np.linspace(0.08 + width / 2, 0.98 - width / 2, node_count)
    return centers, width


def _pipeline(results: dict[str, Any]) -> plt.Figure:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.0, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lane_y = {"C1": 0.78, "C3": 0.50, "C4": 0.20}
    for condition in ("C1", "C3", "C4"):
        nodes = PIPELINE_LABELS[condition].split(" → ")
        centers, width = pipeline_geometry(len(nodes))
        height = 0.13
        y = lane_y[condition]
        ax.text(0.01, y, condition, color=COLORS[condition], weight="bold", va="center", fontsize=10)
        for index, (center, label) in enumerate(zip(centers, nodes)):
            face = "#F4F7FA" if index not in (len(nodes) - 1,) else "#FFFFFF"
            box = FancyBboxPatch(
                (center - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                linewidth=1.4,
                edgecolor=COLORS[condition],
                facecolor=face,
            )
            ax.add_patch(box)
            wrapped = label.replace(" + LoRA ", " + LoRA\n").replace(" (same as C3)", "\n(same as C3)")
            ax.text(center, y, wrapped, ha="center", va="center", fontsize=7.2)
            if index < len(nodes) - 1:
                ax.annotate(
                    "",
                    xy=(centers[index + 1] - width / 2 - 0.008, y),
                    xytext=(center + width / 2 + 0.008, y),
                    arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.1},
                )
    ax.set_title("Matched experimental paths", pad=4)
    fig.tight_layout(pad=0.4)
    return fig


def _overall(results: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    conditions = ("C1", "C3", "C4")
    x = np.arange(2)
    width = 0.23
    for offset, name in enumerate(conditions):
        values = [results["conditions"][name]["accuracy"], results["conditions"][name]["macro_f1"]]
        bars = ax.bar(x + (offset - 1) * width, values, width, color=COLORS[name], label=name)
        _labels(ax, bars)
    ax.set_xticks(x, ["Accuracy", "Macro-F1"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.set_title("Strict performance on the matched ProofWriter subset")
    fig.tight_layout()
    return fig


def _per_class(results: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = ("True", "False", "Unknown")
    x = np.arange(3)
    width = 0.23
    for offset, name in enumerate(("C1", "C3", "C4")):
        values = [results["conditions"][name]["per_class"][label]["f1"] for label in labels]
        bars = ax.bar(x + (offset - 1) * width, values, width, color=COLORS[name], label=name)
        _labels(ax, bars)
    ax.set_xticks(x, labels)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Classwise F1 under strict scoring")
    fig.tight_layout()
    return fig


def _interventions(results: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    trans = results["paired"]["transitions"]
    keys = ("wrong_to_correct", "correct_to_correct", "correct_to_wrong", "wrong_to_wrong")
    labels = ("Wrong→Correct", "Correct→Correct", "Correct→Wrong", "Wrong→Wrong")
    colors = ("#009E73", "#56B4E9", "#D55E00", "#999999")
    bars = ax.bar(np.arange(4), [trans[key] for key in keys], color=colors)
    _labels(ax, bars, percent=False)
    ax.set_xticks(np.arange(4), labels, rotation=12, ha="right")
    ax.set_ylabel("Examples")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Paired effect of routing checkpoint-250 through the bridge")
    fig.tight_layout()
    return fig


def _confusions(results: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    for ax, name in zip(axes, ("C3", "C4")):
        matrix = np.asarray(results["conditions"][name]["confusion_matrix"])
        columns = results["conditions"][name]["confusion_columns"]
        shown = matrix[:, :3] if name == "C3" else matrix
        image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=max(1, shown.max()))
        for row in range(shown.shape[0]):
            for column in range(shown.shape[1]):
                color = "white" if shown[row, column] > shown.max() / 2 else "black"
                ax.text(column, row, str(shown[row, column]), ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(range(shown.shape[1]), columns[: shown.shape[1]], rotation=30, ha="right")
        ax.set_yticks(range(3), ("True", "False", "Unknown"))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(name)
    fig.colorbar(image, ax=axes, shrink=0.78, label="Count")
    return fig


def _depth(results: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    depth = results["depth"]
    x = np.arange(len(depth))
    labels = [f"{row['qdep']}\n(n={row['n']})" for row in depth]
    for name in ("C1", "C3", "C4"):
        values = [row[f"{name}_accuracy"] for row in depth]
        ax.plot(x, values, marker="o", linewidth=2, color=COLORS[name], label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Proof depth (QDep)")
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Accuracy by reasoning depth (observational groups)")
    fig.tight_layout()
    return fig


def _bridge(results: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.9), constrained_layout=True)
    bridge = results["bridge"]
    stages = ("Schema", "Parser", "AST", "Compile", "Executable")
    values = [bridge["schema_valid"], bridge["parser_valid"], bridge["ast_valid"], bridge["compile_success"], bridge["executable"]]
    bars = axes[0].bar(np.arange(5), values, color="#0072B2")
    _labels(axes[0], bars, percent=False)
    axes[0].set_xticks(np.arange(5), stages, rotation=25, ha="right")
    axes[0].set_ylim(0, 210)
    axes[0].set_ylabel("Examples")
    axes[0].set_title("Stage completion")
    errors = bridge["error_categories"]
    labels = ("Correct", "Executable\nwrong", "Malformed/\nschema", "Inconsistent")
    counts = (errors["correct"], errors["incorrect_final_symbolic_answer"], errors["malformed_output"] + errors["schema_invalid"], errors["inconsistent_premises"])
    bars = axes[1].bar(np.arange(4), counts, color=("#009E73", "#D55E00", "#E69F00", "#999999"))
    _labels(axes[1], bars, percent=False)
    axes[1].set_xticks(np.arange(4), labels)
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Final bridge outcomes")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    return fig


def generate_all_figures(results: dict[str, Any], output_dir: Path) -> list[Path]:
    _style()
    builders: dict[str, Callable[[dict[str, Any]], plt.Figure]] = {
        "pipeline_overview": _pipeline,
        "overall_comparison": _overall,
        "per_class_f1": _per_class,
        "intervention_outcomes": _interventions,
        "confusion_matrices": _confusions,
        "reasoning_depth": _depth,
        "bridge_reliability": _bridge,
    }
    written: list[Path] = []
    for stem in FIGURE_STEMS:
        written.extend(_save(builders[stem](results), output_dir, stem))
    return written
