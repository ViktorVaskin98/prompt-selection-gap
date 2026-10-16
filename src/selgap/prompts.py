import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass

SYSTEM_PROMPTS: dict[str, str | None] = {
    "default": None,
    "assistant": "You are a helpful assistant.",
    "tutor": "You are an expert math tutor.",
    "careful": "You are a careful, meticulous problem solver who double-checks every step.",
}

REASONING: dict[str, str] = {
    "none": "",
    "cot": "Let's think step by step.",
    "plan": (
        "First list the known quantities, then solve the problem step by step, "
        "and verify the result."
    ),
}

ANSWER_FORMATS: dict[str, str] = {
    "boxed": "Put your final answer within \\boxed{}.",
    "sentence": "End your response with 'The answer is: <number>'.",
}

SHOTS: tuple[int, ...] = (0, 2)

_CALCULATOR = re.compile(r"<<[^>]*>>")


@dataclass(frozen=True)
class Exemplar:
    question: str
    solution: str
    answer: str


@dataclass(frozen=True)
class CandidatePrompt:
    index: int
    system: str
    reasoning: str
    answer_format: str
    shots: int

    @property
    def name(self) -> str:
        return f"{self.system}|{self.reasoning}|{self.answer_format}|{self.shots}shot"

    def user_message(self, question: str) -> str:
        parts = [question, REASONING[self.reasoning], ANSWER_FORMATS[self.answer_format]]
        return "\n\n".join(part for part in parts if part)

    def messages(self, question: str, exemplars: Sequence[Exemplar]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        system = SYSTEM_PROMPTS[self.system]
        if system is not None:
            messages.append({"role": "system", "content": system})
        for exemplar in exemplars[: self.shots]:
            messages.append({"role": "user", "content": self.user_message(exemplar.question)})
            messages.append(
                {"role": "assistant", "content": render_solution(exemplar, self.answer_format)}
            )
        messages.append({"role": "user", "content": self.user_message(question)})
        return messages


def render_solution(exemplar: Exemplar, answer_format: str) -> str:
    body = _CALCULATOR.sub("", exemplar.solution.split("####")[0]).strip()
    if answer_format == "boxed":
        final = f"The final answer is \\boxed{{{exemplar.answer}}}."
    else:
        final = f"The answer is: {exemplar.answer}"
    return f"{body}\n{final}"


def candidate_grid() -> list[CandidatePrompt]:
    combos = itertools.product(SYSTEM_PROMPTS, REASONING, ANSWER_FORMATS, SHOTS)
    return [
        CandidatePrompt(index=i, system=s, reasoning=r, answer_format=a, shots=k)
        for i, (s, r, a, k) in enumerate(combos)
    ]
