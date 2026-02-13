from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str = "#ffffff",
    edgecolor: str = "#222222",
    linestyle: str = "solid",
) -> None:
    rect = plt.Rectangle(
        (x, y),
        w,
        h,
        fill=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.3,
        linestyle=linestyle,
        joinstyle="round",
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10.2,
        color="#1a1a1a",
        linespacing=1.15,
    )


def arrow(ax, x1: float, y1: float, x2: float, y2: float, text: str | None = None) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.35, color="#2b2b2b", shrinkA=2, shrinkB=2),
    )
    if text:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.015,
            text,
            ha="center",
            va="center",
            fontsize=9.0,
            color="#202020",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f7f7f7", edgecolor="#cfcfcf", linewidth=0.8),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig_dir", type=str, default="source/figs_compute_v2")
    args = ap.parse_args()

    out_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A compact, paper-friendly layout (caption lives in LaTeX; avoid redundant in-figure titles).
    plt.figure(figsize=(7.6, 3.4))
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    c_base = "#f3f7ff"
    c_ttt = "#f3fff5"
    c_arc = "#fff7e8"
    c_demo = "#f8f4ff"
    c_sdft = "#f0f4ff"
    c_upd = "#f2fff2"

    # Top row: per-instance discovery (ephemeral).
    box(ax, 0.03, 0.62, 0.20, 0.27, "Base LM\n$\\theta_0$\n(mHC)", facecolor=c_base)
    box(ax, 0.27, 0.62, 0.27, 0.27, "TTT-Discover\nLoRA $\\Delta\\theta$\n(ephemeral)", facecolor=c_ttt)
    box(ax, 0.58, 0.62, 0.18, 0.27, "Archive\n(states, rewards)", facecolor=c_arc)
    box(ax, 0.79, 0.62, 0.18, 0.27, "Best\nsolution", facecolor=c_demo)

    arrow(ax, 0.23, 0.76, 0.27, 0.76, "freeze $\\theta_0$")
    arrow(ax, 0.54, 0.76, 0.58, 0.76)
    arrow(ax, 0.76, 0.76, 0.79, 0.76)

    # Separator cue: ephemeral vs persistent.
    ax.plot([0.02, 0.98], [0.54, 0.54], linestyle=(0, (4, 3)), color="#aaaaaa", linewidth=1.0)
    ax.text(0.03, 0.56, "Per-instance (ephemeral)", fontsize=9.0, color="#555555", va="bottom")
    ax.text(0.03, 0.50, "Persistent update (checkpoint)", fontsize=9.0, color="#555555", va="top")

    # Bottom row: consolidation (persistent).
    box(ax, 0.18, 0.16, 0.26, 0.26, "Demos\n{prompt, demo}", facecolor=c_demo)
    box(ax, 0.48, 0.16, 0.22, 0.26, "SDFT\n(reverse KL)\nEMA teacher", facecolor=c_sdft)
    box(ax, 0.74, 0.16, 0.23, 0.26, "Updated\ncheckpoint\n$\\theta_{\\mathrm{cont}}$", facecolor=c_upd)

    arrow(ax, 0.88, 0.62, 0.31, 0.42, "convert")
    arrow(ax, 0.44, 0.29, 0.48, 0.29)
    arrow(ax, 0.70, 0.29, 0.74, 0.29)

    plt.tight_layout(pad=0.25)
    plt.savefig(out_dir / "F1_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_dir / "F1_pipeline.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close()


if __name__ == "__main__":
    main()
