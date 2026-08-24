"""Budget vector construction: layer-wise KV cache budgets -> budgets/*.json.

Schema (see axiom-project-plan.md Part 3):
    {"name", "layer_budgets", "total_budget", "mean_budget", "provenance"}

Strategies that depend on measure.py's LMBA output (zigzag, mass-optimal,
lambda-interpolations) are added once measure.py exists.
"""

import json
from pathlib import Path

BUDGETS_DIR = Path(__file__).resolve().parent.parent / "budgets"


def _round_preserving_sum(weights: list[float], total: int) -> list[int]:
    """Scale weights to sum exactly to `total`, largest-remainder rounding."""
    scaled = [w / sum(weights) * total for w in weights]
    floored = [int(x) for x in scaled]
    remainder = total - sum(floored)
    order = sorted(range(len(weights)), key=lambda i: scaled[i] - floored[i], reverse=True)
    for i in order[:remainder]:
        floored[i] += 1
    return floored


def uniform_budget(num_layers: int, total_budget: int) -> list[int]:
    return _round_preserving_sum([1.0] * num_layers, total_budget)


def pyramid_budget(num_layers: int, total_budget: int) -> list[int]:
    """Placeholder decreasing arithmetic schedule for smoke-testing run_eval.py.

    Not PyramidKV's actual formula -- that gets reproduced from the upstream
    repo when baselines are validated. Provenance is labeled accordingly.
    """
    weights = [float(num_layers - i) for i in range(num_layers)]
    return _round_preserving_sum(weights, total_budget)


def random_budget(num_layers: int, total_budget: int, seed: int) -> list[int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.1, 1.0, size=num_layers).tolist()
    return _round_preserving_sum(weights, total_budget)


def save_budget(name: str, layer_budgets: list[int], total_budget: int, provenance: str) -> Path:
    assert sum(layer_budgets) == total_budget, (
        f"{name}: sum(layer_budgets)={sum(layer_budgets)} != total_budget={total_budget}"
    )
    BUDGETS_DIR.mkdir(exist_ok=True)
    payload = {
        "name": name,
        "layer_budgets": layer_budgets,
        "total_budget": total_budget,
        "mean_budget": total_budget / len(layer_budgets),
        "provenance": provenance,
    }
    out_path = BUDGETS_DIR / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def load_budget(name: str) -> dict:
    payload = json.loads((BUDGETS_DIR / f"{name}.json").read_text())
    assert sum(payload["layer_budgets"]) == payload["total_budget"], (
        f"{name}: budget file failed sum invariant on load"
    )
    return payload


if __name__ == "__main__":
    NUM_LAYERS = 32  # Llama-3.1-8B-Instruct
    TOTAL_BUDGET = 8192  # mean_budget=256

    save_budget(
        "test_uniform",
        uniform_budget(NUM_LAYERS, TOTAL_BUDGET),
        TOTAL_BUDGET,
        "uniform, hand-written Day-1 smoke test",
    )
    save_budget(
        "test_fake_pyramid",
        pyramid_budget(NUM_LAYERS, TOTAL_BUDGET),
        TOTAL_BUDGET,
        "placeholder decreasing schedule, NOT the real PyramidKV formula -- Day-1 smoke test only",
    )
    save_budget(
        "test_random",
        random_budget(NUM_LAYERS, TOTAL_BUDGET, seed=42),
        TOTAL_BUDGET,
        "random weights, seed=42, Day-1 smoke test",
    )
    print(f"wrote 3 test budget files to {BUDGETS_DIR}")
