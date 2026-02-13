from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ISO_FMT = "%Y-%m-%dT%H:%M:%S+00:00"
PHASE_ORDER = ["A", "B", "C", "D", "E", "F"]


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, ISO_FMT)


def collect_minutes_and_steps(state_path: Path, phases: set[str]) -> tuple[dict[str, float], list[tuple[float, str, str]]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = state.get("steps", {})
    phase_minutes = defaultdict(float)
    step_durations: list[tuple[float, str, str]] = []
    for step_name, step in steps.items():
        phase = str(step.get("phase", "?"))
        if phase not in phases:
            continue
        started = parse_ts(step.get("started_at"))
        finished = parse_ts(step.get("finished_at"))
        if started and finished:
            minutes = max(0.0, (finished - started).total_seconds() / 60.0)
            phase_minutes[phase] += minutes
            step_durations.append((minutes, str(step_name), phase))
    return dict(phase_minutes), step_durations


def pretty_step(label: str) -> str:
    # Make the y-axis readable without losing traceability.
    # Example: C4_ttt_main_seed0 -> C4 ttt_main s0
    parts = label.split("_")
    if len(parts) >= 3 and parts[-1].startswith("seed"):
        seed = parts[-1].replace("seed", "s")
        core = "_".join(parts[1:-1])
        return f"{parts[0]} {core} {seed}"
    return label.replace("_", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_ab", default="runs/orchestrator/state.json")
    ap.add_argument("--state_cf", default="runs/orchestrator_fast_v1/state.json")
    ap.add_argument("--fig_dir", default="source/figs_compute_v2")
    args = ap.parse_args()

    minutes_ab, steps_ab = collect_minutes_and_steps(Path(args.state_ab), {"A", "B"})
    minutes_cf, steps_cf = collect_minutes_and_steps(Path(args.state_cf), {"C", "D", "E", "F"})

    phase_minutes = defaultdict(float)
    for phase in PHASE_ORDER:
        if phase in minutes_ab:
            phase_minutes[phase] += minutes_ab[phase]
        if phase in minutes_cf:
            phase_minutes[phase] += minutes_cf[phase]

    step_durations = steps_ab + steps_cf

    out_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))

    # Panel A: runtime footprint by phase.
    ax = axes[0]
    phases = [p for p in PHASE_ORDER if p in phase_minutes]
    mins = [phase_minutes.get(p, 0.0) for p in phases]

    phase_color = {
        "A": "#8c8c8c",
        "B": "#6a3d9a",
        "C": "#355c9c",
        "D": "#e17b12",
        "E": "#3f9852",
        "F": "#b83c46",
    }
    bars = ax.bar(phases, mins, color=[phase_color.get(p, "#777777") for p in phases], width=0.70)
    ax.set_title("A. Runtime Footprint by Phase (A→F composite)")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Wall-clock minutes")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.8,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.96,
        "A/B from runs/orchestrator\nC–F from runs/orchestrator_fast_v1",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#cccccc", alpha=0.85),
    )

    # Panel B: longest step bottlenecks.
    ax = axes[1]
    top_steps = sorted(step_durations, key=lambda x: x[0], reverse=True)[:8]
    labels = [pretty_step(name) for _, name, _ in top_steps]
    counts = [mins for mins, _, _ in top_steps]
    colors = [phase_color.get(phase, "#777777") for _, _, phase in top_steps]

    ypos = np.arange(len(labels))
    ax.barh(ypos, counts, color=colors, height=0.58)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock minutes")
    ax.set_title("B. Longest Execution Steps")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)

    for y, c in zip(ypos, counts):
        ax.text(c + 0.10, y, f"{c:.1f}", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "F4_execution_summary.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "F4_execution_summary.png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
