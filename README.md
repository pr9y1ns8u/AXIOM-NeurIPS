# AXIOM

Instrument for testing whether attention-mass-based layer-wise KV cache budget
allocation (e.g. ZigZagKV) predicts downstream output quality, on
Llama-3.1-8B-Instruct + LongBench. No training; a forked eviction hook plus a
measurement/allocation/eval pipeline. See `axiom-project-plan.md` for the full plan.

## Run

```sh
uv sync
python src/allocate.py            # writes hand-written test budgets to budgets/
python src/run_eval.py <budget>   # writes budgets/{name}.json -> results/{name}__{task}.json
jupyter lab analysis.ipynb        # reads results/, emits every figure and table
```

## Results

- `budgets/` — per-layer budget vectors (input contract, see Part 3 of the plan).
- `results/` — per-sample scores + manifest per run (output contract, see Part 3).
- `figs/` — figures emitted by `analysis.ipynb`. Nothing is hand-copied into the paper.
- `EXPERIMENTS.md` — append-only log, one entry per run.
