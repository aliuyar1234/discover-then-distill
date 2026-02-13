from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Example line:
# [step 3/12] beta=7.345 loss=68.8754 best_reward=0.800000
STEP_RE = re.compile(
    r"\[step (\d+)/(\d+)\] beta=([0-9.\-eE]+) loss=([0-9.\-eE]+) best_reward=([0-9.\-eE]+)"
)

# Naming used in the included run artifacts.
COND_MAIN_S0 = "ttt_main_seed0"
COND_MAIN_S1 = "ttt_main_seed1"
COND_REUSE0 = "ttt_reuse0_seed0"
COND_ADAPT0 = "ttt_adapt0_seed0"
COND_KL0 = "ttt_kl0_seed0"
COND_EXPECTED = "ttt_expected_seed0"

COND_LABELS: Dict[str, str] = {
    COND_MAIN_S0: "Main TTT s0",
    COND_MAIN_S1: "Main TTT s1",
    COND_REUSE0: "Reuse off",
    COND_ADAPT0: "Adaptive-beta off",
    COND_KL0: "KL shaping off",
    COND_EXPECTED: "Expected objective",
}

FAMILY_ORDER = ("add", "sort", "rev")


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_rewards(path: Path) -> List[float]:
    return [float(r["best_reward"]) for r in read_jsonl(path)]


def exact_match_rate(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(1 for v in values if abs(v - 1.0) < 1e-12) / len(values))


def mean_sem(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, 0.0
    sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    return mean, sem


def bootstrap_delta(
    a: List[float],
    b: List[float],
    n_boot: int = 5000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b) by resampling within each list."""
    rng = random.Random(seed)
    la = len(a)
    lb = len(b)
    if la == 0 or lb == 0:
        return float("nan"), float("nan"), float("nan")

    diffs: List[float] = []
    for _ in range(n_boot):
        sa = [a[rng.randrange(la)] for _ in range(la)]
        sb = [b[rng.randrange(lb)] for _ in range(lb)]
        diffs.append(float(np.mean(sa) - np.mean(sb)))

    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    point = float(np.mean(a) - np.mean(b))
    return point, lo, hi


def parse_ttt_stdout_steps(stdout_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return xs, mean, sem, n_per_step for best_reward traces aggregated over tasks."""
    step_rewards: Dict[int, List[float]] = defaultdict(list)
    with open(stdout_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            best_r = float(m.group(5))
            step_rewards[step].append(best_r)

    if not step_rewards:
        return None

    xs = np.array(sorted(step_rewards.keys()), dtype=int)
    means = np.array([float(np.mean(step_rewards[x])) for x in xs], dtype=float)
    stds = np.array([float(np.std(step_rewards[x], ddof=1)) if len(step_rewards[x]) > 1 else 0.0 for x in xs], dtype=float)
    ns = np.array([len(step_rewards[x]) for x in xs], dtype=float)
    sems = np.where(ns > 0, stds / np.sqrt(ns), 0.0)
    return xs, means, sems, ns


def task_family(task_id: str) -> str:
    return task_id.split("_")[0]


def per_task_mean(rows: List[Dict]) -> Dict[str, float]:
    task2vals: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        task2vals[str(row["task_id"])].append(float(row["best_reward"]))
    return {k: float(np.mean(v)) for k, v in task2vals.items()}


def plot_f2(runs_dir: Path, fig_dir: Path) -> None:
    # Paper-friendly: moderate title sizes, consistent grids.
    plt.rcParams.update({
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
    })

    # Wider canvas so the condition legend sits outside Panel A instead of covering curves.
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0))

    # ---- Panel A: discovery trajectories ----
    ax = axes[0]

    # Main seeds first (same color, different linestyle).
    main_color = "#2c7fb8"
    for cond, ls, alpha in [
        (COND_MAIN_S0, "-", 1.0),
        (COND_MAIN_S1, "--", 0.95),
    ]:
        parsed = parse_ttt_stdout_steps(runs_dir / cond / "stdout.log")
        if parsed is None:
            continue
        xs, ys, sem, n = parsed
        ax.plot(xs, ys, label=COND_LABELS[cond], linewidth=2.2, color=main_color, linestyle=ls, alpha=alpha)
        ax.fill_between(xs, ys - 1.96 * sem, ys + 1.96 * sem, color=main_color, alpha=0.10)

    # Ablations (seed 0 controls).
    ablation_colors = {
        COND_REUSE0: "#31a354",
        COND_ADAPT0: "#756bb1",
        COND_KL0: "#de2d26",
        COND_EXPECTED: "#636363",
    }
    for cond in [COND_REUSE0, COND_ADAPT0, COND_KL0, COND_EXPECTED]:
        parsed = parse_ttt_stdout_steps(runs_dir / cond / "stdout.log")
        if parsed is None:
            continue
        xs, ys, sem, n = parsed
        color = ablation_colors[cond]
        ax.plot(xs, ys, label=COND_LABELS[cond], linewidth=2.0, color=color)
        ax.fill_between(xs, ys - 1.96 * sem, ys + 1.96 * sem, color=color, alpha=0.08)

    ax.set_xlabel("TTT step")
    ax.set_ylabel("Mean best reward")
    ax.set_title("A. Discovery trajectories (mean +/- 95% CI over tasks)")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_ylim(0.70, 1.01)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=8.0,
        title="Conditions",
        title_fontsize=8.5,
    )

    ax.text(
        0.02,
        0.98,
        "Held-out set; 12 steps x 8 rollouts (N=96)\nAblations are seed-0 controls",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#cccccc", alpha=0.85),
    )

    # ---- Panel B: main comparison ----
    ax = axes[1]
    base = read_rewards(runs_dir / "base_bestofn_seed0" / "summary.jsonl") + read_rewards(
        runs_dir / "base_bestofn_seed1" / "summary.jsonl"
    )
    ttt = read_rewards(runs_dir / COND_MAIN_S0 / "summary.jsonl") + read_rewards(runs_dir / COND_MAIN_S1 / "summary.jsonl")
    sdft = read_rewards(runs_dir / "sdft500_bestofn_seed0" / "summary.jsonl") + read_rewards(
        runs_dir / "sdft500_bestofn_seed1" / "summary.jsonl"
    )

    groups = [base, ttt, sdft]
    labels = ["Base\nbest-of-N", "Main TTT\n(held-out)", "Post-SDFT\nbest-of-N"]
    means, sems = zip(*(mean_sem(g) for g in groups))
    ems = [exact_match_rate(g) for g in groups]

    x = np.arange(len(labels))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    bars = ax.bar(x, means, yerr=sems, capsize=4, width=0.62, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean best reward")
    ax.set_title("B. Main held-out comparison (mean +/- SEM)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_ylim(0.77, 0.87)

    for b, sem, em in zip(bars, sems, ems):
        y = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y + sem + 0.004,
            f"EM {100.0 * em:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    # Bootstrap deltas (for reference; the paper table uses these numbers).
    d_ttt, lo_ttt, hi_ttt = bootstrap_delta(ttt, base, seed=0)
    d_sdft, lo_sdft, hi_sdft = bootstrap_delta(sdft, base, seed=0)
    ax.text(
        0.02,
        0.04,
        f"Bootstrap delta reward vs base (95% CI):\nTTT: {d_ttt:+.4f} [{lo_ttt:+.4f}, {hi_ttt:+.4f}]\nPost-SDFT: {d_sdft:+.4f} [{lo_sdft:+.4f}, {hi_sdft:+.4f}]",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#cccccc", alpha=0.85),
    )

    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(fig_dir / "F2_reward_vs_steps.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(fig_dir / "F2_reward_vs_steps.png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_f3(runs_dir: Path, fig_dir: Path, heldout_tasks: Path | None = None) -> None:
    base_rows = read_jsonl(runs_dir / "base_bestofn_seed0" / "summary.jsonl") + read_jsonl(
        runs_dir / "base_bestofn_seed1" / "summary.jsonl"
    )
    ttt_rows = read_jsonl(runs_dir / COND_MAIN_S0 / "summary.jsonl") + read_jsonl(runs_dir / COND_MAIN_S1 / "summary.jsonl")
    sdft_rows = read_jsonl(runs_dir / "sdft500_bestofn_seed0" / "summary.jsonl") + read_jsonl(
        runs_dir / "sdft500_bestofn_seed1" / "summary.jsonl"
    )

    base_task = per_task_mean(base_rows)
    ttt_task = per_task_mean(ttt_rows)
    sdft_task = per_task_mean(sdft_rows)

    common_ttt = sorted(set(base_task).intersection(ttt_task))
    common_sdft = sorted(set(base_task).intersection(sdft_task))

    delta_ttt = [ttt_task[t] - base_task[t] for t in common_ttt]
    delta_sdft = [sdft_task[t] - base_task[t] for t in common_sdft]

    # Mean shifts on paired per-task deltas (same held-out tasks).
    d_ttt = float(np.mean(delta_ttt)) if delta_ttt else float("nan")
    d_sdft = float(np.mean(delta_sdft)) if delta_sdft else float("nan")

    # Family-wise deltas.
    fam2vals: Dict[str, List[float]] = defaultdict(list)
    for task_id in common_sdft:
        fam2vals[task_family(task_id)].append(sdft_task[task_id] - base_task[task_id])
    fam_delta = [float(np.mean(fam2vals.get(f, [float("nan")]))) for f in FAMILY_ORDER]

    fam_counts = None
    if heldout_tasks is not None and heldout_tasks.exists():
        tasks = read_jsonl(heldout_tasks)
        counts = defaultdict(int)
        for t in tasks:
            counts[task_family(str(t["task_id"]))] += 1
        fam_counts = {f: int(counts.get(f, 0)) for f in FAMILY_ORDER}

    # Retention summary.
    retention_json = runs_dir / "retention_ppl.json"
    delta_ppl = None
    ppl_base = None
    ppl_updated = None
    if retention_json.exists():
        retention = json.loads(retention_json.read_text(encoding="utf-8-sig"))
        delta_ppl = float(retention["delta_ppl"])
        ppl_base = float(retention["ppl_base"])
        ppl_updated = float(retention["ppl_updated"])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8))

    # ---- Panel A: distribution of shifts ----
    ax = axes[0]
    xpos = np.arange(2)
    deltas = [delta_ttt, delta_sdft]
    colors = ["#4c78a8", "#54a24b"]

    bxp = ax.boxplot(
        deltas,
        positions=xpos,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, color in zip(bxp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)

    # light jittered points
    for i, (vals, color) in enumerate(zip(deltas, colors)):
        rng = np.random.default_rng(100 + i)
        jitter = rng.uniform(-0.09, 0.09, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, alpha=0.18, color=color, edgecolors="none")

    # Mean markers (paired per-task deltas).
    ax.scatter(xpos, [d_ttt, d_sdft], s=75, color="#1f1f1f", zorder=3)

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)

    pct_ttt = 100.0 * float(np.mean([v > 0.0 for v in delta_ttt]))
    pct_sdft = 100.0 * float(np.mean([v > 0.0 for v in delta_sdft]))
    ax.set_xticks(xpos, [f"TTT - Base\n({pct_ttt:.1f}% > 0)", f"Post-SDFT - Base\n({pct_sdft:.1f}% > 0)"])
    ax.set_ylabel("Per-task reward delta")
    ax.set_title("A. Task-level shifts (distribution + mean)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    ax.text(
        0.02,
        0.04,
        f"Mean shift (paired): TTT {d_ttt:+.4f}\nMean shift (paired): Post-SDFT {d_sdft:+.4f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#cccccc", alpha=0.85),
    )

    # ---- Panel B: family-wise consolidation shift ----
    ax = axes[1]
    xpos = np.arange(len(FAMILY_ORDER))
    fam_colors = ["#d64a4a" if y < 0 else "#2e9f45" for y in fam_delta]
    bars = ax.bar(xpos, fam_delta, color=fam_colors, width=0.62, edgecolor="#222222", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)

    xt = []
    for f in FAMILY_ORDER:
        if fam_counts is None:
            xt.append(f.upper())
        else:
            xt.append(f"{f.upper()} (n={fam_counts.get(f, 0)})")
    ax.set_xticks(xpos, xt)

    ax.set_ylabel("Post-SDFT - Base (mean reward)")
    ax.set_title("B. Family-wise consolidation shift")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    for b in bars:
        y = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y + (0.0015 if y >= 0 else -0.0015),
            f"{y:+.3f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    if delta_ppl is not None and ppl_base is not None and ppl_updated is not None:
        ax.text(
            0.02,
            0.95,
            f"Retention PPL: {ppl_base:.1f} -> {ppl_updated:.1f} (delta {delta_ppl:+.3f})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#cccccc", alpha=0.85),
        )

    fig.tight_layout()
    fig.savefig(fig_dir / "F3_gain_vs_retention.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(fig_dir / "F3_gain_vs_retention.png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs/suite_fast_v1")
    ap.add_argument("--fig_dir", type=str, default="paper/figs_compute_v2")
    ap.add_argument("--heldout_tasks", type=str, default="data/discovery/heldout.jsonl")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    fig_dir = Path(args.fig_dir)
    heldout_tasks = Path(args.heldout_tasks) if args.heldout_tasks else None

    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_f2(runs_dir, fig_dir)
    plot_f3(runs_dir, fig_dir, heldout_tasks=heldout_tasks)


if __name__ == "__main__":
    main()
