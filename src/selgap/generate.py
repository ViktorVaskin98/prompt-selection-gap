from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selgap.answers import extract_prediction, is_correct
from selgap.data import Problem
from selgap.prompts import CandidatePrompt, Exemplar


@dataclass(frozen=True)
class Record:
    qid: str
    candidate: int
    text: str
    prediction: str | None
    correct: bool
    prompt_tokens: int
    generated_tokens: int
    hit_max_tokens: bool


class GreedyGenerator:
    def __init__(self, model_id: str, dtype: torch.dtype, device: str = "cuda") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device).eval()

    def run(
        self,
        candidate: CandidatePrompt,
        problems: Sequence[Problem],
        exemplars: Sequence[Exemplar],
        batch_size: int,
        max_new_tokens: int,
    ) -> Iterator[Record]:
        for start in range(0, len(problems), batch_size):
            batch = problems[start : start + batch_size]
            yield from self._complete(candidate, batch, exemplars, max_new_tokens)

    @torch.inference_mode()
    def _complete(
        self,
        candidate: CandidatePrompt,
        batch: Sequence[Problem],
        exemplars: Sequence[Exemplar],
        max_new_tokens: int,
    ) -> list[Record]:
        prompts = [
            self.tokenizer.apply_chat_template(
                candidate.messages(p.question, exemplars),
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        output = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output[:, encoded["input_ids"].shape[1] :]
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        stop_ids = {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}
        records: list[Record] = []
        for row, tokens in enumerate(generated.tolist()):
            kept = [t for t in tokens if t not in stop_ids]
            text = self.tokenizer.decode(kept, skip_special_tokens=True)
            prediction = extract_prediction(text)
            records.append(
                Record(
                    qid=batch[row].qid,
                    candidate=candidate.index,
                    text=text,
                    prediction=prediction,
                    correct=is_correct(prediction, batch[row].gold),
                    prompt_tokens=prompt_lengths[row],
                    generated_tokens=len(kept),
                    hit_max_tokens=len(kept) >= max_new_tokens,
                )
            )
        return records
