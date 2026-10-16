from selgap.prompts import Exemplar, candidate_grid, render_solution

EXEMPLAR = Exemplar(
    question="Tom has 3 apples and buys 4 more. How many apples?",
    solution="He has 3 + 4 = <<3+4=7>>7 apples.\n#### 7",
    answer="7",
)


def test_grid_is_full_factorial_with_stable_indices() -> None:
    grid = candidate_grid()
    assert len(grid) == 48
    assert [c.index for c in grid] == list(range(48))
    assert len({c.name for c in grid}) == 48


def test_render_solution_strips_calculator_and_marker() -> None:
    rendered = render_solution(EXEMPLAR, "boxed")
    assert "<<" not in rendered and "####" not in rendered
    assert rendered.endswith("\\boxed{7}.")
    assert render_solution(EXEMPLAR, "sentence").endswith("The answer is: 7")


def test_messages_follow_factors() -> None:
    by_name = {c.name: c for c in candidate_grid()}
    plain = by_name["default|none|boxed|0shot"].messages("Q?", [EXEMPLAR, EXEMPLAR])
    assert plain == [{"role": "user", "content": "Q?\n\nPut your final answer within \\boxed{}."}]
    shot = by_name["tutor|cot|sentence|2shot"].messages("Q?", [EXEMPLAR, EXEMPLAR])
    assert [m["role"] for m in shot] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert shot[-1]["content"].startswith("Q?\n\nLet's think step by step.")
