# AXIOM @ NeurIPS 2026 — project context

**Deadline:** 29 Aug 2026 · 4 pages excl. references · non-archival · double-blind
**Hardware:** RTX 4050 laptop (6 GB VRAM), Colab Pro for a second model
**Scope:** solo, five days

---

## The claim, in two sentences

Every method that splits a KV cache budget across layers decides the split using
**attention mass** — keep enough tokens per layer to retain ~90% of the attention
weight. We show that attention mass is the wrong thing to measure: what damages
the output is not *how much* you dropped, but *whether what you dropped was
different from what you kept*.

That's the whole paper. If you can't say it this plainly, don't write it.

---

## The one piece of math

An attention head outputs a weighted average of value vectors:

    o = Σ_i a_i · v_i           where a is a probability distribution over n cached tokens

Evict down to a set `S` of size `B`. Softmax renormalises over what survives, so
the new output is `õ = Σ_{i∈S} (a_i / m_S) · v_i`, where `m_S = Σ_{i∈S} a_i` is
the retained mass.

Split the original output into what was kept and what was dropped:

    o = m_S · õ  +  (1 − m_S) · μ         μ = centroid of the evicted values

Rearranged:

    ‖o − õ‖  =  (1 − m_S) · ‖μ − õ‖
                └────┬────┘   └───┬───┘
              how much you    how DIFFERENT
                 dropped       it was

**Exact. Not a bound.** One line of algebra.

Every layer-allocation method in the literature measures the left factor and
ignores the right one. Two layers can drop identical attention mass and suffer
completely different output damage, because the second factor varies
independently.

That is the mechanism. Everything else is measurement.

### The falsifiable prediction (state this BEFORE showing results)

> Attention-mass loss will correlate weakly with actual output error across
> layers and budgets, because the geometric factor `‖μ − õ‖` varies
> independently of the retained mass.

Falsified if the correlation is strong (say ρ > 0.9).

---

## The one experiment

For every layer and every budget `B`, compute two numbers from the same cached
tensors:

| | |
|---|---|
| `D_att(B) = −log m_B` | how much attention mass was dropped |
| `D_out(B) = ‖o − õ(B)‖ / ‖o‖` | how far the output actually moved |

Scatter one against the other, colour by layer depth, take a Spearman
correlation. **That correlation number is the paper's thesis.**

### Why it's cheap

Sort the value rows by descending attention and take prefix sums
`P(B) = Σ_{i≤B} a_(i) v_(i)`. Then `õ(B) = P(B)/m_B` and `o = P(n)`, so both
curves come out for **every budget at once**, in closed form, at cost O(nd) per
head. One forward pass total. Nothing is ever evicted; no text is generated.

---

## What we are explicitly NOT doing

An earlier version of this plan had all of the following. **All of it is cut.**
If you find yourself reaching for any of it, stop — it is not in scope.

- Water-filling, KKT conditions, equal-marginal-distortion allocators
- The reduction of ZigZagKV's MBA to a point on a distortion curve
- Comparing five allocation schedules against each other
- Submodularity under query uncertainty
- Any end-to-end eviction run
- LongBench, RULER, needle-in-a-haystack, or any benchmark
- Forking KVCache-Factory or any compression framework
- `measure.py` and `allocate.py` (superseded by `probe.py`)

The paper does not propose a better allocator. It argues the signal everyone
allocates on is mis-specified. Those are different papers; we are writing the
second one.

---

## Background you need (and only this much)

Eviction methods make two independent decisions:

**Scoring** — which tokens survive. H2O uses accumulated attention, SnapKV uses
attention from a recent observation window, StreamingLLM keeps sinks plus recent
tokens.

**Allocation** — how many tokens each layer gets. Uniform was the unargued
default. PyramidKV imposes an arithmetic decreasing schedule. ZigZagKV allocates
proportionally to a per-layer attention-mass requirement, with a floor patched on.

None of these methods change model weights. They monkeypatch the attention
forward, run `topk` on a score, and `index_select` the K and V tensors down to
`B` rows. That is the entire intervention.

PyramidKV, ZigZagKV and Ada-KV all reuse SnapKV's scoring and argue only about
allocation — and all three use attention mass as the allocation signal. That
shared signal is our target.

---

## Running it

```bash
pip install "torch>=2.3" "transformers>=4.44" numpy scipy matplotlib
python probe.py --model meta-llama/Llama-3.2-1B --ctx 4096 --probes 8
python probe.py --model Qwen/Qwen2.5-1.5B --ctx 4096 --probes 8
```

Outputs `scatter.pdf`, `curves.npz`, and the correlation printed to stdout.
Runs in about 15 minutes on the 4050.

---

## Invariants — assert these, don't hope

- `D_out[:, -1] < 1e-3` — at full budget `õ = o` exactly, so the error must
  vanish. **This is the strongest check on the identity.** If it fails, the V
  cache is misaligned or GQA head expansion is wrong.
- `D_att[:, -1] < 1e-3` — same, for the mass side.
- Attention rows sum to 1.0 ± 1e-3.
- `n_q_heads % n_kv_heads == 0` before expanding V.

---

## Known breakages

**Cache accessor.** `cache.layers[l].values` vs `cache.value_cache[l]` vs
`cache[l][1]` depending on transformers version. `probe.py` tries all three; if
all fail, print `type(cache)` and `dir(cache)` rather than guessing.

**Attention weights.** The model must load with `attn_implementation="eager"` —
SDPA and FlashAttention don't return weights. But eager materialises `n×n`, so
prefill must be **chunked** with `output_attentions=False`, and attentions
requested only at decode where query length is 1.

**GQA.** Llama-3.2 and Qwen2.5 have fewer KV heads than query heads. V must be
repeated along the head axis before pairing with attention rows. Getting this
wrong produces plausible-looking garbage.

**Precision.** Cast to float32 before cumsum and log. fp16 cumsum over 4K terms
loses precision exactly in the tail that matters.

---

## Outcome branches

Run the correlation before writing anything.

| ρ | Reading | Action |
|---|---|---|
| weak (< 0.7) | the proxy doesn't track the damage | paper as framed |
| middling (0.7–0.9) | proxy is partial | same paper, softer language, lead with per-layer spread |
| strong (> 0.9) | proxy is better than its justification | honest reframe: "the field's heuristic works for a reason it never stated" |

All three are publishable. Do **not** tune the budget grid, probe count, context
length, or head-aggregation rule until a preferred number appears. Report what
comes out.

---

## Paper structure (4 pages)

| § | Space | Content |
|---|---|---|
| 1 Introduction | 0.5 p | scoring vs allocation; every allocator uses attention mass; nobody checked whether it predicts output damage |
| 2 The identity | 0.8 p | the factorisation, the two factors, the prediction |
| 3 Method | 0.5 p | what's measured and why it needs one forward pass |
| 4 Results | 1.7 p | the scatter, the correlation, per-layer breakdown, second model |
| 5 Limitations | 0.5 p | see below |

---

## Limitations (name these; don't half-fix them)

- Single query per probe step; eviction commits before future queries arrive.
- Measured along the full-KV path — no trajectory drift from actual generation.
- Per-head error, not end-to-end task accuracy.
- Layers treated independently; evicting early does reshape later layers.
- Two small models, not 7B+.

Naming these honestly is worth more than half-attempting any of them.

---

## Deferred to revision (post-submission)

OpenReview accepts PDF replacements until the deadline, so submit early and
revise. In rough order of likely reviewer demand:

1. End-to-end next-token KL under real simultaneous eviction
2. A 7–8B model on Colab L4
3. An allocator built on `D_out` instead of `D_att`, to show the gap is actionable
4. Cross-layer interaction measurement
5. A downstream task

---

## Aggregation convention

Arithmetic mean over query heads and probe steps. This is a modelling choice and
must be stated in the paper. Max-over-heads is the natural alternative; it goes
in the appendix if there's time, which there won't be.

---

## If you read nothing else

1. `‖o − õ‖ = (1 − m_S) · ‖μ − õ‖`. Dropped mass times displacement.
2. The field measures the first factor only.
3. One forward pass gives both curves for all budgets via prefix sums.
4. The Spearman correlation between them is the result.
5. Report whatever number comes out.