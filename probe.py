"""
probe.py -- the entire experiment, one file.

Question:  does attention-mass loss predict output error?

For every layer and every budget B we compute two numbers from the SAME
cached tensors:

    D_att(B) = -log m_B                     how much attention mass we dropped
    D_out(B) = ||o - o_tilde(B)|| / ||o||   how far the output actually moved

Then we scatter them against each other and take a rank correlation.
Weak correlation => the signal the field allocates on does not track the
damage it is supposed to predict.

No eviction is ever performed. No text is generated. No benchmark is run.

    python probe.py --model meta-llama/Llama-3.2-1B
"""

import argparse
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load the variables from the .env file
load_dotenv()

# Grab the token from your environment
hf_token = os.getenv("HF_TOKEN")

# Log into Hugging Face
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in .env file.")


def get_values(cache, layer):
    """V for one layer, shape (n_kv_heads, seq, head_dim). API has moved around."""
    for get in (
        lambda: cache.layers[layer].values,
        lambda: cache.value_cache[layer],
        lambda: cache[layer][1],
    ):
        try:
            v = get()
            if v is not None:
                return v[0]
        except (AttributeError, IndexError, TypeError):
            continue
    raise RuntimeError(f"cannot read V from {type(cache)}; check transformers version")


def two_curves(a, v, grid):
    """
    a: (n,)   attention row, sums to 1
    v: (n, d) value vectors
    Returns D_att, D_out, D_disp on the budget grid.

    Sort by attention, then take BOTH prefix and suffix sums. The suffix sums
    matter: the evicted centroid mu = sum_{i>B} a_i v_i / (1 - m_B) must be
    computed from a directly-accumulated tail, never as (o - P[B])/(1 - m[B]).
    That subtraction loses all significant digits once m_B approaches 1, which
    happens early in attention-sink heads and makes the ratio explode.
    """
    order = np.argsort(-a)
    a_s, v_s = a[order], v[order]
    w = a_s[:, None] * v_s                    # a_i * v_i, sorted

    m = np.cumsum(a_s)                        # retained mass at budget B
    P = np.cumsum(w, axis=0)                  # retained weighted sum
    o = P[-1]                                 # full attention output

    # suffix sums: tail_mass[B-1] = sum of a_s[B:], Q[B-1] = sum of w[B:]
    tail_mass = np.concatenate([np.cumsum(a_s[::-1])[::-1][1:], [0.0]])
    Q = np.concatenate([np.cumsum(w[::-1], axis=0)[::-1][1:], np.zeros((1, v.shape[1]))])

    # scale for normalisation: typical value-vector magnitude in this head.
    # Using ||o|| instead is unstable -- o can be near zero when value vectors
    # cancel, which would inflate every ratio in that head.
    vscale = np.sqrt(np.mean(np.sum(v_s ** 2, axis=1))) + 1e-12

    i = grid - 1
    m_i = np.clip(m[i], 1e-12, 1.0)
    t_i = tail_mass[i]

    D_att = -np.log(m_i)
    o_tilde = P[i] / m_i[:, None]
    D_out = np.linalg.norm(o - o_tilde, axis=1) / (np.linalg.norm(o) + 1e-12)

    # displacement ||mu - o_tilde||, in units of the value scale.
    # Undefined where nothing is evicted -- mark NaN rather than dividing.
    ok = t_i > 1e-9
    D_disp = np.full(len(i), np.nan)
    if ok.any():
        mu = Q[i][ok] / t_i[ok][:, None]
        D_disp[ok] = np.linalg.norm(mu - o_tilde[ok], axis=1) / vscale
    return D_att, D_out, D_disp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--probes", type=int, default=8)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="auto = bfloat16 on CUDA, float32 on CPU. fp16 overflows at long "
                        "context (logits exceed 65504 -> inf -> softmax gives NaN).")
    p.add_argument("--text", default=None)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
    else:
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"device={dev}  dtype={dt}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dt,
        attn_implementation="eager",     # SDPA will not return attention weights
    ).to(dev).eval()

    raw = open(args.text).read() if args.text else "The mitochondrion is a double-membrane-bound organelle. " * 4000
    ids = tok(raw, return_tensors="pt").input_ids[:, : args.ctx].to(dev)

    # Prefill in chunks with attentions OFF, so eager never builds a big n x n.
    cache = None
    with torch.no_grad():
        for s in range(0, ids.shape[1] - 1, args.chunk):
            out = model(ids[:, s : min(s + args.chunk, ids.shape[1] - 1)],
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if not torch.isfinite(out.logits).all():
                raise RuntimeError(
                    f"non-finite logits during prefill at token offset {s}. "
                    f"dtype={dt}. If this is float16, rerun with --dtype bf16 "
                    f"(fp16 overflows on attention logits at long context).")
    cur = ids[:, -1:]

    L = model.config.num_hidden_layers
    grid = np.unique(np.round(np.geomspace(1, args.ctx - 1, 64)).astype(int))
    A = np.zeros((L, len(grid)))
    O = np.zeros((L, len(grid)))
    G = np.zeros((L, len(grid)))     # displacement factor ||mu - o_tilde||

    # Probe steps. Query length is 1, so attentions are only (heads, n) -- tiny.
    with torch.no_grad():
        for _ in range(args.probes):
            out = model(cur, past_key_values=cache, use_cache=True, output_attentions=True)
            cache = out.past_key_values

            for l in range(L):
                att = out.attentions[l][0, :, -1, :].float().cpu().numpy()   # (H_q, n)
                V = get_values(cache, l).float().cpu().numpy()               # (H_kv, n, d)

                assert att.shape[0] % V.shape[0] == 0, "GQA head mismatch"
                V = np.repeat(V, att.shape[0] // V.shape[0], axis=0)         # expand for GQA

                # The real validity check: these must be probability distributions.
                # If this fails we are not looking at post-softmax attention.
                if not np.isfinite(att).all():
                    raise RuntimeError(
                        f"layer {l}: attention contains NaN/inf ({(~np.isfinite(att)).sum()} "
                        f"of {att.size} entries), dtype={dt}, n={att.shape[1]}. "
                        f"fp16 overflows on attention logits at long context; rerun with "
                        f"--dtype bf16, or --dtype fp32 if bf16 is unavailable.")
                if not np.isfinite(V).all():
                    raise RuntimeError(f"layer {l}: value cache contains NaN/inf, dtype={dt}")
                row_sums = att.sum(axis=1)
                assert np.abs(row_sums - 1.0).max() < 1e-2, (
                    f"layer {l}: attention rows do not sum to 1 "
                    f"(min {row_sums.min():.4f}, max {row_sums.max():.4f})")

                n = att.shape[1]
                g = grid[grid <= n]
                pad = len(grid) - len(g)

                per_head = [two_curves(att[h], V[h, :n], g) for h in range(att.shape[0])]
                da = np.mean([c[0] for c in per_head], axis=0)
                do = np.mean([c[1] for c in per_head], axis=0)
                with np.errstate(invalid="ignore"):
                    dg = np.nanmean([c[2] for c in per_head], axis=0)
                if pad:
                    da = np.concatenate([da, np.zeros(pad)])
                    do = np.concatenate([do, np.zeros(pad)])
                    dg = np.concatenate([dg, np.full(pad, np.nan)])
                A[l] += da
                O[l] += do
                G[l] += dg

            cur = out.logits[:, -1:].argmax(-1)
            n_final = n
            del out
            if dev == "cuda":
                torch.cuda.empty_cache()

    A /= args.probes
    O /= args.probes
    G /= args.probes

    # ---- invariants ----
    # The largest grid point is ctx-1, but the cache grows by one token per
    # probe step, so grid[-1] is NOT full budget -- it means "drop the
    # (n_final - grid[-1]) least-attended tokens". Under uniform attention
    # those tokens alone carry (n_final - grid[-1]) / n_final of the mass, so
    # the tolerance has to scale with context length and probe count, not sit
    # at a fixed 1e-3 calibrated on a short smoke test.
    n_dropped = max(n_final - int(grid[-1]), 1)
    uniform_tail = n_dropped / n_final
    tol = 2.0 * uniform_tail          # generous: uniform attention is the worst case

    tail_att = A[:, -1]
    tail_out = O[:, -1]
    print(f"\nTail check at B={grid[-1]} (cache reached n={n_final}, so {n_dropped} tokens dropped)")
    print(f"  uniform-attention tail mass would be {uniform_tail:.2e}; tolerance {tol:.2e}")
    print(f"  max D_att at largest budget: {tail_att.max():.2e}  (layer {int(tail_att.argmax())})")
    print(f"  max D_out at largest budget: {tail_out.max():.2e}  (layer {int(tail_out.argmax())})")

    # Displacement is a distance between two centroids of value vectors,
    # normalised by the typical value magnitude. It is bounded by a small
    # multiple of 1 by construction. If it is not, the tail computation has
    # gone unstable again and the numbers are meaningless.
    gmax = np.nanmax(G)
    print(f"\nmax displacement (units of value scale): {gmax:.3f}")
    assert gmax < 50, (
        f"displacement reached {gmax:.3e} -- this is a centroid distance and "
        f"cannot legitimately be this large. The tail-sum computation is unstable.")

    assert tail_att.max() < tol, (
        f"tail attention mass {tail_att.max():.2e} exceeds even the uniform-attention "
        f"bound {tol:.2e} -- this is not diffuse attention, something is wrong")
    assert tail_out.max() < 0.1, (
        f"dropping {n_dropped} least-attended tokens moved the output by "
        f"{tail_out.max():.2e} relative -- implausible, check V alignment")

    # Diffuseness by depth: how concentrated is each layer's attention?
    # A layer whose tail mass approaches the uniform bound is diffuse; one far
    # below it is concentrated. This is the PyramidKV observation, measured.
    print("\n  per-layer tail mass (fraction of the uniform bound):")
    ratios = tail_att / uniform_tail
    for l in range(0, L, max(L // 8, 1)):
        bar = "#" * int(40 * min(ratios[l], 1.0))
        print(f"    layer {l:>2}  {ratios[l]:>6.3f}  {bar}")

    # ---- the number ----
    # Allocation ranks LAYERS at a MATCHED budget, so that is the correlation
    # that matters. A pooled correlation over all (layer, budget) pairs is
    # inflated by the shared monotone trend in B and must not be the headline.
    #
    # per_budget stays the SAME LENGTH as grid, always -- NaNs (from degenerate
    # ties, most likely at very small B where several layers retain identical
    # mass) are kept in place rather than dropped, so grid[i] and per_budget[i]
    # never drift out of alignment. Use `valid` to mask when needed.
    per_budget = np.array([spearmanr(A[:, b], O[:, b]).statistic for b in range(len(grid))])
    valid = ~np.isnan(per_budget)
    rho = float(np.nanmean(per_budget))
    print(f"\nHEADLINE  mean cross-layer Spearman at matched budget: {rho:.3f}")
    if valid.any():
        print(f"          range over budgets: [{np.nanmin(per_budget):.3f}, {np.nanmax(per_budget):.3f}]")
    if (~valid).any():
        print(f"          ({(~valid).sum()}/{len(grid)} budgets had degenerate ties and were excluded)")

    pooled, _ = spearmanr(A.ravel(), O.ravel())
    print(f"(pooled over all pairs, inflated by the shared B trend: {pooled:.3f} -- do not report as headline)")

    # The explanatory variable: is the displacement factor layer-invariant?
    cv = np.nanmean(np.nanstd(G, axis=0) / (np.nanmean(G, axis=0) + 1e-12))
    print(f"\nDisplacement factor spread across layers (mean CV): {cv:.3f}")
    print("  small CV  -> displacement is layer-invariant -> attention mass is a valid ranking signal")
    print("  large CV  -> displacement varies by layer     -> attention mass cannot rank layers correctly")

    # ---- figure 1: pooled scatter (all layers, all budgets together) ----
    # NOTE: this pools across budgets, so its visual tightness is dominated by
    # the shared trend against B (more eviction -> more damage, for everyone).
    # It is NOT a picture of the headline claim -- label it as the pooled view.
    fig, ax = plt.subplots(figsize=(5, 4))
    for l in range(L):
        ax.scatter(A[l], O[l], s=6, color=plt.cm.viridis(l / max(L - 1, 1)), alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"attention-mass distortion  $-\log m_B$")
    ax.set_ylabel(r"relative output error  $\|o-\tilde o\|/\|o\|$")
    ax.set_title(rf"pooled over layers & budgets, $\rho={pooled:.2f}$" + "\n(dominated by shared trend vs. B -- not the headline)", fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax, label="layer depth")
    fig.savefig("figs_llama/scatter_pooled.pdf", bbox_inches="tight")

    # ---- figure 2: the actual headline claim, visualized ----
    # correlation of D_att vs D_out ACROSS LAYERS, computed separately at each
    # fixed budget. This is what an allocator actually needs: at one budget,
    # does attention-mass loss rank layers the same way output error does?
    # Plot only the valid (non-degenerate) budgets, using the mask -- grid and
    # per_budget are always the same length so this indexing is always correct,
    # even if the invalid entries aren't a clean prefix/suffix.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(grid[valid], per_budget[valid], marker="o", ms=3, lw=1, color="C0")
    ax.axhline(rho, color="C1", ls="--", lw=1, label=f"mean = {rho:.2f}")
    ax.set_xscale("log")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("budget $B$")
    ax.set_ylabel(r"cross-layer Spearman$(D_{att}, D_{out})$ at fixed $B$")
    ax.set_title("does attention mass rank layers correctly? (this is the claim)", fontsize=8)
    ax.legend(fontsize=8)
    fig.savefig("figs_llama/rho_vs_budget.pdf", bbox_inches="tight")


    np.savez("figs_llama/curves.npz", grid=grid, D_att=A, D_out=O, D_disp=G,
             model=args.model, rho=rho, pooled_rho=pooled,
             per_budget_rho=per_budget, per_budget_valid=valid)
    print("wrote figs_llama/scatter_pooled.pdf, figs_llama/rho_vs_budget.pdf, and figs_llama/curves.npz")


if __name__ == "__main__":
    main()