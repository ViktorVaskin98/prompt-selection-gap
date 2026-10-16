import argparse
from pathlib import Path

from selgap.plots import plot_candidates, plot_optimism
from selgap.results import candidate_table, load_correctness
from selgap.selection import simulate, summarize

SEARCH_SIZES = [25, 50, 100, 200, 400]
CANDIDATE_COUNTS = [2, 4, 8, 16, 32, 48]


def main() -> None:
    parser = argparse.ArgumentParser(description="Selection gap of automated prompt search")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--splits", type=int, default=1000)
    parser.add_argument("--holdout", type=int, default=300)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()

    correct, qids, indices = load_correctness(args.run_dirs)
    print(f"matrix {correct.shape[0]} candidates x {correct.shape[1]} problems")
    candidates = candidate_table(correct, indices)
    sizes = [n for n in SEARCH_SIZES if n <= correct.shape[1] - args.holdout]
    counts = [k for k in CANDIDATE_COUNTS if k <= correct.shape[0]]
    summary = summarize(
        simulate(correct, sizes, counts, n_splits=args.splits, holdout_size=args.holdout)
    )

    args.results.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.results / "candidates.csv", index=False)
    summary.to_csv(args.results / "selection_summary.csv", index=False)
    plot_candidates(candidates, args.figures / "candidates.png")
    plot_optimism(summary, args.figures / "selection_gap.png")
    print(candidates.round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
