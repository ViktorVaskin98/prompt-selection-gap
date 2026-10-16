import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from selgap.prompts import candidate_grid


def load_correctness(run_dirs: Sequence[Path]) -> tuple[np.ndarray, list[str], list[int]]:
    with (run_dirs[0] / "problems.jsonl").open(encoding="utf-8") as source:
        qids = [json.loads(line)["qid"] for line in source]
    position = {qid: i for i, qid in enumerate(qids)}
    rows: dict[int, np.ndarray] = {}
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("cand_*.jsonl")):
            index = int(path.stem.removeprefix("cand_"))
            row = np.full(len(qids), np.nan)
            with path.open(encoding="utf-8") as source:
                for record in map(json.loads, source):
                    row[position[record["qid"]]] = float(record["correct"])
            if np.isnan(row).any():
                raise ValueError(f"{path} does not cover the whole pool")
            rows[index] = row
    indices = sorted(rows)
    return np.vstack([rows[i] for i in indices]), qids, indices


def candidate_table(correct: np.ndarray, indices: Sequence[int]) -> pd.DataFrame:
    grid = {c.index: c for c in candidate_grid()}
    n = correct.shape[1]
    accuracy = correct.mean(axis=1)
    return pd.DataFrame(
        {
            "index": list(indices),
            "name": [grid[i].name for i in indices],
            "system": [grid[i].system for i in indices],
            "reasoning": [grid[i].reasoning for i in indices],
            "answer_format": [grid[i].answer_format for i in indices],
            "shots": [grid[i].shots for i in indices],
            "accuracy": accuracy,
            "ci": 1.96 * np.sqrt(accuracy * (1 - accuracy) / n),
        }
    ).sort_values("accuracy", ascending=False)
