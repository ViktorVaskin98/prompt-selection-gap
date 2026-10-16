from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#c9a227", "#8d5a97"]


def _finish(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_optimism(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, sharey=False)
    sizes = sorted(summary["n"].unique())
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sizes)}
    for n in sizes:
        for protocol, style in [("naive", "-"), ("two-stage", "--")]:
            curve = summary[(summary["n"] == n) & (summary["protocol"] == protocol)]
            curve = curve.sort_values("N")
            axes[0].errorbar(
                curve["N"],
                100 * curve["optimism"],
                yerr=100 * curve["optimism_ci"],
                linestyle=style,
                marker="o",
                markersize=3,
                capsize=2,
                color=colors[n],
                label=f"n={n}, {protocol}",
            )
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Candidates searched, N (log scale)")
    axes[0].set_ylabel("Optimism, pp (reported − holdout)")
    axes[0].set_title("How much the winner's score is inflated")
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    for n in sizes:
        for protocol, style in [("naive", "-"), ("two-stage", "--")]:
            curve = summary[(summary["n"] == n) & (summary["protocol"] == protocol)]
            curve = curve.sort_values("N")
            axes[1].plot(
                curve["N"],
                100 * curve["regret"],
                linestyle=style,
                marker="o",
                markersize=3,
                color=colors[n],
            )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Candidates searched, N (log scale)")
    axes[1].set_ylabel("Regret, pp (best holdout − picked holdout)")
    axes[1].set_title("How much worse the winner is than the true best")
    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    _finish(fig, out_path)


def plot_candidates(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=160)
    ordered = table.sort_values("accuracy")
    colors = [PALETTE[0] if shots == 0 else PALETTE[1] for shots in ordered["shots"]]
    ax.barh(range(len(ordered)), 100 * ordered["accuracy"], xerr=100 * ordered["ci"], color=colors)
    ax.set_yticks(range(len(ordered)), ordered["name"], fontsize=5.5)
    ax.set_xlabel("Accuracy on the 800-problem pool, % (95% CI)")
    ax.set_title("48 prompt candidates for Qwen2.5-0.5B on GSM8K (blue: 0-shot, orange: 2-shot)")
    ax.set_xlim(left=max(0, 100 * ordered["accuracy"].min() - 10))
    ax.spines[["top", "right"]].set_visible(False)
    _finish(fig, out_path)
