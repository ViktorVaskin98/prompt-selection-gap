import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from selgap.data import load_exemplars, load_pool
from selgap.generate import GreedyGenerator
from selgap.prompts import candidate_grid

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass(frozen=True)
class RunPlan:
    model_id: str
    dtype: str
    n_problems: int
    pool_seed: int
    exemplar_seed: int
    max_new_tokens: int
    batch_size: int
    candidate_start: int
    candidate_end: int

    @classmethod
    def from_json(cls, path: Path) -> "RunPlan":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def run_candidates(plan: RunPlan, out_dir: Path) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_pool(plan.n_problems, plan.pool_seed)
    exemplars = load_exemplars(2, plan.exemplar_seed)
    (out_dir / "problems.jsonl").write_text(
        "".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in problems),
        encoding="utf-8",
    )
    generator = GreedyGenerator(plan.model_id, DTYPES[plan.dtype])
    summary: dict[str, dict] = {}
    for candidate in candidate_grid()[plan.candidate_start : plan.candidate_end]:
        started = time.time()
        n_correct = n_total = n_truncated = 0
        path = out_dir / f"cand_{candidate.index:02d}.jsonl"
        with path.open("w", encoding="utf-8") as sink:
            for record in generator.run(
                candidate, problems, exemplars, plan.batch_size, plan.max_new_tokens
            ):
                sink.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                n_total += 1
                n_correct += record.correct
                n_truncated += record.hit_max_tokens
        stats = {
            "name": candidate.name,
            "accuracy": n_correct / n_total,
            "truncated_share": n_truncated / n_total,
            "seconds": round(time.time() - started, 1),
        }
        summary[str(candidate.index)] = stats
        print(f"[cand {candidate.index:02d}] {stats}", flush=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
