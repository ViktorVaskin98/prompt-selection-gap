from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from selgap.answers import extract_gold
from selgap.prompts import Exemplar


@dataclass(frozen=True)
class Problem:
    qid: str
    question: str
    gold: str


def load_pool(n_problems: int, seed: int) -> list[Problem]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    order = np.random.default_rng(seed).permutation(len(dataset))[:n_problems]
    return [
        Problem(
            qid=f"gsm8k-test-{int(i)}",
            question=dataset[int(i)]["question"],
            gold=extract_gold(dataset[int(i)]["answer"]),
        )
        for i in order
    ]


def load_exemplars(n_exemplars: int, seed: int) -> list[Exemplar]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    order = np.random.default_rng(seed).permutation(len(dataset))[:n_exemplars]
    return [
        Exemplar(
            question=dataset[int(i)]["question"],
            solution=dataset[int(i)]["answer"],
            answer=extract_gold(dataset[int(i)]["answer"]),
        )
        for i in order
    ]
