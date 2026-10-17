# PLAN — prompt-selection-gap

A pilot on selection bias in automated prompt and agent search.

## Question
Automated prompt search (and, more broadly, search over agent configurations) picks the candidate
that scored best on a fixed search set of tasks. How inflated is the picked candidate's score — the
"cost of selection" — as a function of the number of candidates searched *N* and the search-set size
*n*, and which selection protocol reduces it at the same compute budget?

## Setup
- Qwen2.5-0.5B-Instruct, greedy, up to 512 new tokens, Kaggle T4.
- Pool: 800 GSM8K test problems (seed 1). Every candidate is run once on the whole pool → a
  48 × 800 correctness matrix. All selection experiments are offline simulations on it (1000 random
  splits of the pool into a search set and a held-out set).
- Candidates: 48 prompts = 4 system prompts × 3 reasoning instructions × 2 answer formats ×
  {0, 2} worked examples from GSM8K train.
- Protocols at an equal budget of `N · n` evaluations:
  - **naive** — argmax on a search set of *n* problems;
  - **two-stage** — all *N* candidates on *n*/2 problems → top 3 → re-evaluation on fresh problems
    with the remaining budget → pick and report from the second stage.
- Metrics: optimism = acc_search(picked) − acc_holdout(picked);
  regret = best acc_holdout among the *N* − acc_holdout(picked).

## Outcome (2026-10-16, 18:45 MSK)
1 — confirmed (10.0 pp); 2 — partly (only for N ≥ 8); 3 — confirmed (24.6 pp). Details in the
[README](README.md).

## Predictions (fixed on 2026-10-16, before any data)
1. Optimism grows with *N* and falls with *n*; at n = 50 and N = 48 it exceeds 3 pp.
2. At an equal budget, two-stage has lower optimism than naive for every n ≥ 50.
3. The spread of true accuracy across the 48 prompts is at least 5 pp, so selection matters at all.
