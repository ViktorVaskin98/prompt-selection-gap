import numpy as np

from selgap.selection import naive_selection, simulate, summarize, two_stage_selection


def _noise_candidates(n_candidates: int, n_problems: int, p: float, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).random((n_candidates, n_problems)) < p).astype(float)


def test_identical_candidates_have_no_optimism() -> None:
    row = (np.random.default_rng(0).random(600) < 0.5).astype(float)
    correct = np.tile(row, (8, 1))
    table = summarize(simulate(correct, [50], [8], n_splits=400))
    naive = table[table["protocol"] == "naive"].iloc[0]
    assert abs(naive["optimism"]) < 0.02
    assert naive["regret"] == 0.0


def test_optimism_grows_with_candidates_under_pure_noise() -> None:
    correct = _noise_candidates(32, 600, 0.5, seed=1)
    table = summarize(simulate(correct, [50], [2, 32], n_splits=300))
    naive = table[table["protocol"] == "naive"].set_index("N")["optimism"]
    assert naive.loc[32] > naive.loc[2] > 0.0
    assert naive.loc[32] > 0.08


def test_two_stage_reduces_optimism_under_pure_noise() -> None:
    correct = _noise_candidates(32, 600, 0.5, seed=2)
    table = summarize(simulate(correct, [50], [32], n_splits=300)).set_index("protocol")
    assert table.loc["two-stage", "optimism"] < table.loc["naive", "optimism"]


def test_clearly_better_candidate_is_found() -> None:
    rng = np.random.default_rng(3)
    correct = _noise_candidates(6, 400, 0.4, seed=3)
    correct[4] = (rng.random(400) < 0.9).astype(float)
    candidates = np.arange(6)
    order = rng.permutation(400)
    holdout, material = order[:100], order[100:]
    naive = naive_selection(correct, candidates, material[:200], holdout, rng)
    staged = two_stage_selection(correct, candidates, 200, material, holdout, 3, rng)
    assert naive.holdout == staged.holdout == correct[4, holdout].mean()
