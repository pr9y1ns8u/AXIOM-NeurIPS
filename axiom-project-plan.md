# AXIOM @ NeurIPS 2026 — Project Plan

**Paper:** *Attention Mass Is the Wrong Currency: On the Failure of Layer-wise KV Cache Budget Allocation*
**Deadline:** 29 Aug 2026, 11:59 PM UTC (= 05:29 AM IST, Sun 30 Aug) · 4 pages excl. references · double-blind · non-archival
**Team:** Zaid, Banik

---

## Part 0 — What we are actually building

| | |
|---|---|
| Training / fine-tuning | **None.** Zero gradient steps. |
| Model reimplementation | **None.** HuggingFace Llama-3.1-8B-Instruct, unmodified weights. |
| What we modify | The **eviction hook** in a forked repo: add per-layer logging, and accept an arbitrary per-layer budget vector instead of a hardcoded schedule. |
| What we reimplement | **ZigZagKV's Eq. 4–6 only** (~30 lines). Falls out free once we have LMBA. |
| What we benchmark | LongBench, as *evidence for a claim*, not as a leaderboard entry. |

The deliverable is an **instrument**: a pipeline that takes a per-layer budget vector as input and emits (a) attention-mass statistics and (b) downstream accuracy with confidence intervals. Everything else is analysis on top of that.

### Hardware split
- **4050 / 6 GB** — code development against `Llama-3.2-1B-Instruct`, analysis, plotting, writing. Never runs the real sweep.
- **Colab Pro (A100 40 GB or L4 24 GB)** — all 8B inference. T4 (16 GB) is insufficient.

---

## Part 1 — Reading list, in access order

### Tier 0 — Before writing a line of code (Sun 23 – Mon 24)

| # | Paper | arXiv / venue | What to extract |
|---|---|---|---|
| 1 | **Ada-KV** | 2407.11550 · NeurIPS 2025 | **Read Thm 3.1, Thm 3.2, Appendix A.9 in full.** This is *our objective*. Extract: exact form of the L1 bound; the softmax-renormalization step in the proof; the claim that global top-k is the optimal allocation. Also grab their explicit "future work: extend head-wise allocation across layers" sentence — cite it as the gap we address. |
| 2 | **ZigZagKV** | 2412.09036 | Already read. Re-extract precisely: MBA/LMBA definitions (§3), Eq. 4–6 (for reimplementation), **Tables 3 & 4** (attention loss vs hidden-state loss — our Fig. 1b), **Figure 6** (bounded-budget ablation — our §4 prediction). |
| 3 | **LAVa** | 2509.09754 · Findings EMNLP 2025 | Nearest competitor. Read Theorem 1 and their layer-budget derivation. We must be able to state in one sentence how our claim differs from theirs. Non-negotiable citation. |
| 4 | **AATC** | 2608.14191 (14 Aug 2026) | Nine days old. Check their attention-aware distortion factorization against anything we write down. Skim proof only. |

### Tier 1 — Skim Monday–Tuesday, cite in §5

| # | Paper | ID | Use |
|---|---|---|---|
| 5 | **CriticalKV** | 2502.03805 | Attention weights insufficient — value states + projection matter. The *scoring-level* version of our claim. Must differentiate. |
| 6 | **PyramidKV** | 2406.02069 | Target #1. Arithmetic schedule, no stated objective. |
| 7 | **CAKE** | ICLR 2025 | Third unstated-objective layer method. One clause in §5. |
| 8 | **RateQuant** | 2605.06675 | The **AM/GM-of-sensitivities** result: a general condition for *when non-uniform allocation can help at all*. Potentially lets us predict flatness analytically. |
| 9 | **RDKV** | 2605.08317 | Rate–distortion + water-filling for KV. Cite so nobody thinks we claim the framing. |
| 10 | **PolyKV** | 2606.15157 | Allocation has a "conditional effect"; uniform more reliable at small budgets. **Contradicts ZigZagKV's small-budget claim** — flag this contradiction, it is exactly AXIOM's stated interest. |
| 11 | **"Ablation, Statistical Inference, and Validation for KV-Cache Compression"** | 2607.09683 | Water-filling collapses to uniform. Same shape of null, quantization domain. Also a model for reporting nulls statistically. |
| 12 | **"Rethinking KV Cache Compression for LLM Serving"** | 2503.24000 | Negative-sample analysis: benchmark averages hide per-sample fragility. Justifies why 0.1-pt LongBench deltas are meaningless. |
| 13 | **KV Cache Survey (TMLR 05/2025)** | 2412.19442 | Our taxonomy source. §4.2.1 Layer-wise / §4.2.2 Head-wise budget allocation. |
| 14 | **Keep the Cost Down (COLM 2024)** | 2407.18003 | Cite twice only: (a) §6.2 evaluation metrics list contains *no* output-fidelity measure; (b) Keyformer note — post-eviction attention renormalizes and becomes uneven. Relevant to our derivation's correctness. |

### Tier 2 — Only if time permits

| # | Paper | ID | Use |
|---|---|---|---|
| 15 | SqueezeAttention | 2404.04793 | Layer budgets from an *output-side* signal. If it beats ZigZagKV that supports our thesis directly. |
| 16 | "Physics of KV Cache Compression" | 2603.01426 | Retention vs accessibility vs utilization — useful vocabulary for §6. |
| 17 | Bouthillier et al., "Accounting for Variance in ML Benchmarks" | 2103.03098 | Bootstrap CI methodology. |
| 18 | Devoto et al. (L2-norm eviction) | 2406.11430 | Attention-free scoring signal; supporting evidence the field already distrusts attention scores. |

---

## Part 2 — Codebases and data

> **Verify every URL before cloning.** Repos move. If a link 404s, search the paper title on GitHub.

### Primary base repo
**PyramidKV / KVCache-Factory** — `github.com/Zefan-Cai/PyramidKV`

Chosen because it already contains: SnapKV, PyramidKV, H2O, StreamingLLM; LongBench eval scripts; and — decisively — **existing plumbing for non-uniform per-layer budgets** (the pyramid schedule). Making it accept an arbitrary budget vector is a small edit, not an architecture change. This single fact is worth ~2 days.

### Reference / cross-check
- `github.com/FFY0/AdaKV` — official Ada-KV. Read `adaptive_snapkv/` for how they compute the L1 bound in code. **Do not fork**; read only.
- `github.com/NVIDIA/kvpress` — hook-based, cleaner abstractions. Use as a reference implementation if PyramidKV's monkey-patching becomes unworkable. Fallback base, not primary.
- **ZigZagKV** — no confirmed official repo. **Reimplement Eq. 4–6.** Trivial once `measure.py` produces LMBA.

### Paper indexes (for §5 completeness check on Friday)
- `github.com/October2001/Awesome-KV-Cache-Compression`
- `github.com/TreeAI-Lab/Awesome-KV-Cache-Management`
- `github.com/zcli-charlie/Awesome-KV-Cache`

### Models
- `meta-llama/Llama-3.1-8B-Instruct` — all reported results. **Pin the revision hash.**
- `meta-llama/Llama-3.2-1B-Instruct` — local dev on the 4050. Never reported.

### Data
- **LongBench** — `THUDM/LongBench` (HF datasets)
  - *Measurement:* `2wikimqa`, 200 samples. Chosen to be directly comparable to ZigZagKV Figs. 2–3 and Tables 3–4.
  - *Sweep:* 6 tasks, one per LongBench category — `2wikimqa` (multi-doc QA), `qasper` (single-doc QA), `gov_report` (summarization), `trec` (few-shot), `passage_retrieval_en` (synthetic), `lcc` (code). Full category coverage is a defensible scope claim.
- **NIAH** — `github.com/gkamradt/LLMTest_NeedleInAHaystack`. Stretch goal only.
- **RULER** — out of scope. Say so in Limitations.

---

## Part 3 — Repo structure and the interface contract

```
axiom-kv/
├── README.md              # 10 lines: what, how to run, where results land
├── EXPERIMENTS.md         # append-only log. one entry per run. never edited.
├── env.yml                # pinned versions
├── src/
│   ├── measure.py         # ZAID  — per-layer LMBA/LMBO/deficit/error
│   ├── allocate.py        # ZAID  — budget vectors → budgets/*.json
│   ├── patch.py           # ZAID  — fork of pyramidkv monkeypatch + logging + vector input
│   ├── run_eval.py        # BANIK — budgets/*.json → results/*.json
│   └── stats.py           # BANIK — bootstrap CIs, correlations
├── budgets/               # INTERFACE — Zaid writes, Banik reads
├── results/               # INTERFACE — Banik writes, both read
├── analysis.ipynb         # BANIK — reads results/, emits every figure
└── figs/
```

### The interface contract — read this twice

The single decision that makes two-person parallel work possible in six days.

**`budgets/{name}.json`**
```json
{
  "name": "lambda_1.0_B256",
  "layer_budgets": [412, 388, ...],
  "total_budget": 8192,
  "mean_budget": 256,
  "provenance": "mass-optimal, lambda=1.0, LMBA from run 2026-08-24-a"
}
```

**`results/{name}__{task}.json`**
```json
{
  "budget_file": "lambda_1.0_B256",
  "task": "2wikimqa",
  "per_sample_scores": [41.2, 0.0, 38.7, ...],
  "mean": 35.4,
  "n": 200,
  "manifest": {"git_sha": "...", "model_rev": "...", "seed": 42, "timestamp": "..."}
}
```

**Rules:**
1. `run_eval.py` accepts *any* valid budget file and never knows how it was produced. Banik can build and test it against hand-written budget files on **day one**, before Zaid's measurement finishes.
2. `sum(layer_budgets)` must equal `total_budget`. Assert it. Rounding drift here silently invalidates the whole comparison.
3. **`per_sample_scores` is mandatory.** Bootstrap CIs are impossible without it, and a null result without CIs is a desk reject.
4. Nobody hand-copies a number into the paper. `analysis.ipynb` reads `results/` and emits every figure and every table.

---

## Part 4 — Work split

### Zaid — theory, instrumentation, paper

Owns the novel measurements and the argument.

- **Tier 0 reading** (Ada-KV proof especially — you need to be able to defend the renormalization step)
- `patch.py`: fork PyramidKV's monkeypatch. Two changes: per-layer stat logging, arbitrary budget-vector input.
- `measure.py`: paired LMBA / LMBO / deficit / error in **one forward pass**
- `allocate.py`: uniform · pyramid · zigzag · mass-optimal · λ-interpolations → `budgets/`
- §1, §2, §3, §5, §6 of the paper
- **Owns the Monday go/no-go call**

### Banik — harness, baselines, statistics, figures

Owns everything that can proceed without the measurement existing yet.

- **Day 1, before any results exist:** hand-write three budget files (uniform, a fake pyramid, a random vector), build `run_eval.py` against them, confirm end-to-end on one task. This de-risks the entire week.
- Reproduce SnapKV + PyramidKV LongBench baselines. Do they match the published tables? If not we find out Monday, not Thursday.
- Colab session management: checkpoint per task, resume on disconnect, never lose more than one task.
- `stats.py`: bootstrap CIs over `per_sample_scores`, Spearman correlations with CIs
- `analysis.ipynb` → all figures
- §4 of the paper (the sweep section)

**Swap note:** if Banik is strong on the theory, hand him §2 and take the eval harness instead. Whoever writes §2 must have read the Ada-KV proof.

### Sync points
- **Mon evening** — the two correlation numbers. Go/no-go.
- **Tue evening** — end-to-end smoke test passes on one task × one budget.
- **Thu evening** — sweep complete, CIs computed, figures drafted.

---

## Part 5 — Day by day

| Day | Zaid | Banik |
|---|---|---|
| **Sun 23** | Fork repo. Patch logging. Launch measurement overnight. | Env setup, Colab quota check, pin model revision, hand-write 3 test budget files. |
| **Mon 24** | Ada-KV + LAVa proofs. **Compute both correlations.** Go/no-go by evening. | `run_eval.py` working end-to-end. SnapKV/PyramidKV baselines reproduced. |
| **Tue 25** | `allocate.py` → full λ family + fixed points. | Smoke-test the real budget files. **Find the plumbing bug today.** |
| **Wed 26** | Draft §1–§3 against real Fig. 1. | Launch full sweep: 12 configs × 6 tasks. Babysit. |
| **Thu 27** | Draft §5, §6. Related-work completeness pass. | Sweep completes. Bootstrap CIs. All figures. |
| **Fri 28** | **Write all four pages.** | §4 prose. Figure polish. Template compliance. |
| **Sat 29** | Buffer. Anonymization check. Submit. | Buffer. Reproducibility check from clean clone. |

Friday night is the real deadline. Saturday is slack, not schedule.

---

## Part 6 — Documentation discipline

**`EXPERIMENTS.md` is append-only.** One entry per run, written *before* the run starts:

```
## 2026-08-24-a
Who: Zaid
What: LMBA/LMBO measurement, 2wikimqa, n=200, tau in {0.8,0.9,0.95}
Model: Llama-3.1-8B-Instruct @ <rev>
Git: <sha>
Hypothesis: spearman(LMBA, LMBO) < 0.5
Result: [fill after]
```

Never edit an entry. If a run was wrong, add a new entry saying so. When a reviewer asks "how many configurations did you try before this one," you want a truthful answer available.

**Every result file carries a manifest** — git SHA, model revision, seed, timestamp. A number without a manifest is not evidence.

**Failures get logged too.** OOMs, silent NaNs, budget-sum drift. The bug you don't write down is the bug you reintroduce on Thursday.

---

## Part 7 — Decision gates

**Monday evening — the go/no-go.**

- `spearman(LMBA, LMBO)` weak (say < 0.5) → **proceed to the full paper.** The correlation *is* the thesis.
- Strongly correlated → the central mechanism is wrong; no λ-sweep rescues it. **Switch to AXIOM's Grand Challenges track**: 1 page, light review, and the CFP explicitly solicits "contradictions between theory and practice" and "efficiency methods lacking theoretical explanations." Writable Thursday from ZigZagKV's own Tables 3–4 and Fig. 6, with or without our sweep.

Having this fallback is what makes six days safe. We are not betting the week on the experiments landing.

**Wednesday noon — scope gate.** If the sweep is not running cleanly by then, cut to 4 tasks and 2 budgets. A tight result on less data beats a broad result that doesn't finish.

---

## Explicitly out of scope — put these in Limitations

- Oracle allocation minimizing measured output error *(the obvious next experiment; state it as such)*
- Mistral / Qwen — one model only
- Decoding-time eviction — prefill only
- RULER, InfiniteBench
- Head-wise interaction with Ada-KV

Naming these honestly is worth more than half-attempting one of them.
