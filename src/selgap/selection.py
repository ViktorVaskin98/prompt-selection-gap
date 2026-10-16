from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Outcome:
    reported: float
    holdout: float
    best_holdout: float


def _argmax_random_ties(scores: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(scores == scores.max())
    return int(rng.choice(best))


def naive_selection(
    correct: np.ndarray,
    candidates: np.ndarray,
    search: np.ndarray,
    holdout: np.ndarray,
    rng: np.random.Generator,
) -> Outcome:
    search_scores = correct[np.ix_(candidates, search)].mean(axis=1)
    holdout_scores = correct[np.ix_(candidates, holdout)].mean(axis=1)
    pick = _argmax_random_ties(search_scores, rng)
    return Outcome(
        reported=float(search_scores[pick]),
        holdout=float(holdout_scores[pick]),
        best_holdout=float(holdout_scores.max()),
    )


def two_stage_selection(
    correct: np.ndarray,
    candidates: np.ndarray,
    n_per_candidate: int,
    material: np.ndarray,
    holdout: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
) -> Outcome:
    n_first = n_per_candidate // 2
    first = material[:n_first]
    first_scores = correct[np.ix_(candidates, first)].mean(axis=1)
    k = min(top_k, len(candidates))
    shuffled = rng.permutation(len(candidates))
    shortlist = shuffled[np.argsort(-first_scores[shuffled], kind="stable")][:k]
    remaining = len(candidates) * (n_per_candidate - n_first)
    n_second = min(remaining // k, len(material) - n_first)
    second = material[n_first : n_first + n_second]
    second_scores = correct[np.ix_(candidates[shortlist], second)].mean(axis=1)
    local = _argmax_random_ties(second_scores, rng)
    pick = candidates[shortlist[local]]
    holdout_scores = correct[np.ix_(candidates, holdout)].mean(axis=1)
    return Outcome(
        reported=float(second_scores[local]),
        holdout=float(correct[pick, holdout].mean()),
        best_holdout=float(holdout_scores.max()),
    )


def simulate(
    correct: np.ndarray,
    search_sizes: Sequence[int],
    candidate_counts: Sequence[int],
    n_splits: int = 1000,
    holdout_size: int = 300,
    top_k: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_candidates, n_problems = correct.shape
    if max(search_sizes) > n_problems - holdout_size:
        raise ValueError("search set does not fit next to the holdout")
    rows = []
    for split in range(n_splits):
        order = rng.permutation(n_problems)
        holdout, material = order[:holdout_size], order[holdout_size:]
        for n in search_sizes:
            for count in candidate_counts:
                candidates = rng.choice(n_candidates, size=count, replace=False)
                outcomes = {
                    "naive": naive_selection(correct, candidates, material[:n], holdout, rng),
                    "two-stage": two_stage_selection(
                        correct, candidates, n, material, holdout, top_k, rng
                    ),
                }
                for protocol, outcome in outcomes.items():
                    rows.append(
                        {
                            "split": split,
                            "protocol": protocol,
                            "n": n,
                            "N": count,
                            "optimism": outcome.reported - outcome.holdout,
                            "holdout": outcome.holdout,
                            "regret": outcome.best_holdout - outcome.holdout,
                        }
                    )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["protocol", "n", "N"])
    table = grouped[["optimism", "holdout", "regret"]].mean()
    errors = grouped[["optimism", "holdout", "regret"]].sem().mul(1.96).add_suffix("_ci")
    return table.join(errors).reset_index()
