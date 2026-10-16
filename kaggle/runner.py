import sys
from pathlib import Path

CONFIG_NAME = "smoke"


def locate_code_root() -> Path:
    for marker in Path("/kaggle/input").rglob("selgap/__init__.py"):
        return marker.parent.parent.parent
    raise FileNotFoundError("prompt-selection-gap code dataset is not attached")


def main() -> None:
    root = locate_code_root()
    sys.path.insert(0, str(root / "src"))

    import torch

    from selgap.pipeline import RunPlan, run_candidates

    print("code root:", root, flush=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    plan = RunPlan.from_json(root / "configs" / f"{CONFIG_NAME}.json")
    summary = run_candidates(plan, Path("/kaggle/working") / CONFIG_NAME)
    print("REQUIRED_OUTPUT summary", summary, flush=True)


if __name__ == "__main__":
    main()
