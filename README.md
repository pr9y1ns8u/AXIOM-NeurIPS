# AXIOM

Tests whether attention mass dropped during KV cache eviction predicts how far
the attention output actually moves. One exact identity, one forward pass, no
eviction, no benchmark. See `CONTEXT.md` for the full spec: the identity, the
experiment, invariants, and known breakages.

## Run

```sh
uv sync
python probe.py --model meta-llama/Llama-3.2-1B --ctx 4096 --probes 8
python probe.py --model Qwen/Qwen2.5-1.5B --ctx 4096 --probes 8
```

Outputs `figs/scatter_pooled.pdf`, `figs/rho_vs_budget.pdf`, and
`figs/curves.npz`, plus the Spearman correlation printed to stdout. Runs in
about 15 minutes per model on a 6 GB laptop GPU. Re-running overwrites the
previous run's output.

```sh
python analyze.py
```

Reads `figs/curves.npz` and writes `figs/displacement.pdf`, checking whether
attention mass's failure to predict output error is systematic with layer
depth rather than noise.

## Layout

- `probe.py` — the whole instrument: builds a context window, runs chunked
  prefill, probes decode steps for attention weights, computes `D_att(B)` and
  `D_out(B)` for every budget via prefix sums, and plots the correlation.
- `analyze.py` — everything downstream of the GPU run. Reads `figs/curves.npz`
  only, checks whether the displacement factor is layer-invariant, and
  whether attention mass underestimates damage systematically by depth.
- `figs/` — generated output only (`scatter_pooled.pdf`, `rho_vs_budget.pdf`,
  `displacement.pdf`, `curves.npz`), gitignored. Nothing is hand-copied into
  the paper.
- `CONTEXT.md` — the project spec. If code and `CONTEXT.md` disagree,
  `CONTEXT.md` wins.
