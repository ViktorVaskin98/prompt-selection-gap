from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#c9a227", "#8d5a97"]

STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

PROTOCOL_STYLE = {"naive": "-", "two-stage": "--"}


def _finish(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_optimism(summary: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    sizes = sorted(summary["n"].unique())
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sizes)}
    panels = [
        ("optimism", "Optimism, pp (reported − held-out)", "How inflated is the winner's score?"),
        ("regret", "Regret, pp (true best − picked)", "How far is the winner from the true best?"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        for n in sizes:
            for protocol, style in PROTOCOL_STYLE.items():
                curve = summary[(summary["n"] == n) & (summary["protocol"] == protocol)]
                curve = curve.sort_values("N")
                ax.errorbar(
                    curve["N"],
                    100 * curve[metric],
                    yerr=100 * curve[f"{metric}_ci"],
                    linestyle=style,
                    linewidth=2 if protocol == "naive" else 1.6,
                    marker="o" if protocol == "naive" else "s",
                    markersize=5,
                    capsize=2,
                    color=colors[n],
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 48], ["2", "4", "8", "16", "32", "48"])
        ax.set_xlabel("Candidates searched, N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    size_handles = [Patch(color=colors[n], label=f"n = {n}") for n in sizes]
    protocol_handles = [
        Line2D([], [], color="grey", linestyle="-", marker="o", linewidth=2, label="naive"),
        Line2D([], [], color="grey", linestyle="--", marker="s", linewidth=1.6, label="two-stage"),
    ]
    fig.legend(
        handles=size_handles + protocol_handles,
        loc="lower center",
        ncol=len(size_handles) + len(protocol_handles),
        frameon=False,
        title="Search-set size (colour) and selection protocol (line)",
        bbox_to_anchor=(0.5, -0.12),
    )
    _finish(fig, out_path)


def plot_candidates(table: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(STYLE)
    ordered = table.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(9, 13))
    colors = [PALETTE[0] if shots == 0 else PALETTE[1] for shots in ordered["shots"]]
    labels = [
        f"{r.system} · {r.reasoning} · {r.answer_format}" for r in ordered.itertuples(index=False)
    ]
    ax.barh(
        range(len(ordered)),
        100 * ordered["accuracy"],
        xerr=100 * ordered["ci"],
        color=colors,
        error_kw={"elinewidth": 1, "ecolor": "#555555"},
    )
    ax.set_yticks(range(len(ordered)), labels, fontsize=10)
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.set_xlabel("Accuracy on 800 GSM8K problems, % (95% CI)")
    ax.set_title("48 prompts, one model (Qwen2.5-0.5B)")
    ax.set_xlim(left=max(0, 100 * ordered["accuracy"].min() - 8))
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(color=PALETTE[0], label="0-shot"), Patch(color=PALETTE[1], label="2-shot")],
        loc="lower right",
        frameon=False,
        title="system · reasoning · format",
    )
    _finish(fig, out_path)
