# Small-N "judgement" consistency datasets + Gemma sweep

**Date:** 2026-06-04
**Status:** approved (brainstorming) → implementing

## Goal

The repo measures *preference consistency* of LLMs over pairwise comparisons
(`p_self`, `p_reversal`, `p_acyclic`, `p_crossq`, plus μ-decisiveness). So far the
items have been sentiment targets (people, foods, concepts) and the questions have
been sentiment / interest / shape framings.

This adds two **new small-N datasets of a different flavour** — items that are
*tasks/artifacts to be judged*, not concepts to be liked — and asks **evaluative
"which is X-er" questions** about them. We then run the **Gemma-3 series**
(1b/4b/12b/27b-it) through every dataset×question combination and plot the four
consistency probes against model scale.

The interesting scientific question: *are bigger models more self-consistent when
asked to judge which coding problem is harder / which recipe better tests a cook?*
This probes whether "evaluator coherence" (a capability relevant to LLM-as-judge
use) scales the way sentiment coherence does.

## Datasets (N ≈ 40 each, `config/datasets/*.yaml`)

Same schema as existing datasets: `items:` list of strings, plus an optional
`meta:` block carrying provenance (HF source) and per-item tags.

1. **`leetcode_problems.yaml`** — pulled from a HuggingFace leetcode dataset. Each
   item is a concise, self-contained one-line problem statement (title + short
   gloss) so it fits inline in the A/B template and a 1b model can still reason
   about it. `meta` records the HF source and the official Easy/Medium/Hard
   difficulty per item (a free ground-truth axis for a later sanity check, not used
   by the consistency metrics). Selection: ~balanced across difficulty.

2. **`recipes.yaml`** — pulled from a HuggingFace recipes dataset. Each item is a
   short "dish — one-line description". `meta` records the HF source. Selection:
   varied across cuisine / complexity.

Both built by `scripts/build_small_n_datasets.py`, which genuinely pulls from HF
(via the `datasets` library) and logs which source/commit it used. If HF is
unreachable the script errors loudly rather than silently substituting a baked-in
list (we want a *pulled* dataset).

## Questions (6 banks, `config/questions/*.jsonl`)

Three evaluative **constructs**, each as its own bank with a `pos`/`neg` pair
(valence +1 / −1) — exactly the existing `main`/`interest`/`shape` idiom. `pos` is
the primary (matches `compute_four`'s default `primary_qid="pos"`), `neg` is the
framing reversal used for the cross-question probe.

| construct | pos framing | neg framing |
|---|---|---|
| `harder` | which is **harder** | which is **easier** |
| `interesting` | which is **more interesting** | which is **more boring** |
| `applicant` | which better **tests an applicant's ability** | which is **less useful for assessing ability** |

The `applicant` wording is adapted per dataset:
- leetcode → "a software-engineering candidate's coding ability"
- recipes → "a cook's ability in the kitchen"

So there are 6 bank files: `{leetcode,recipes}_{harder,interesting,applicant}.jsonl`.

Each `pos`/`neg` is within the *same* construct, so `p_crossq` measures
**framing-robustness of that construct** (does "A harder than B" agree with the
negation "B easier than A"?), not cross-construct agreement. This is the faithful
"consistency" reading the user asked for.

## Runs

4 Gemma models × 2 datasets × 3 constructs = **24 elicitation runs**, each the full
four-phase pipeline (elo → reverse → triad → cross_question) producing an
`edges.jsonl`. Output tree: `runs/small_n/<dataset>/<construct>/<model>/`.

Phase budgets scaled for N≈40 (defaults were tuned for 500–2000 items): `R=6, m=8`
(near-complete pair coverage of the 780 possible pairs → stable μ and metrics),
`n_reverse=400`, `n_triads=600`, `n_cross=400`. Logit-local elicitation is
deterministic, so re-sampling a pair is wasteful, not informative — these budgets
are sized to *cover* pairs, not to repeat them.

Backend: `local` logit oracle (greedy A/B softmax), bf16, on RunPod GPUs.

### Efficiency / compute

- `scripts/run_small_n_sweep.py` loads each model **once** and loops it over all 6
  banks (reusing one `LocalLogitOracle`), instead of 6 separate `load_model` calls.
- Cross-GPU parallelism: the runner takes `--models` and `--gpu`; we launch two
  processes pinned via `CUDA_VISIBLE_DEVICES`. Split by weight so wall-clock
  balances: **GPU0 → {27b}**, **GPU1 → {12b, 4b, 1b}** (27b's download+load
  dominates; the three smaller models together ≈ its cost).
- No local GPU on this machine → all runs on RunPod. 27b bf16 ≈ 54 GB → each GPU
  needs ≥80 GB (A100/H100-80GB) since we want one whole model per GPU. CUDA-13
  driver required (repo torch is `2.12.0+cu130`; see memory `b200-cu13-torch-works`).

## Consistency plot

`scripts/plot_small_n_consistency.py`:
- Walk `runs/small_n/**/edges.jsonl`, compute the four probes via
  `four_metrics.compute_four` (+ μ-decisiveness via `compute_decis_and_fit`).
- Tidy frame: `(family=Gemma, model, params_b, dataset, construct, metric, value)`.
- Plot the four probes **vs Gemma scale** (x = params_b, log axis), one facet grid
  per dataset, hue/col = construct, with the chance-floor lines (0.5, and 0.75 for
  `p_acyclic`) drawn in. Seaborn, exported PDF (+ PNG) to `results/plots/`.
- Also emit a tidy CSV `results/small_n_consistency.csv` for the record.

## Logging / provenance

Per repo convention: commit all code before the GPU run and record the commit hash
in each run's `metrics.json` (already done by `run_elicitation`). Timestamped pod
log dir. Tarball+gzip the `runs/small_n` tree (per memory `tarball-edges-before-transfer`)
before scp back, and upload the full log tarball to HF `arcadia-impact`
(`scripts/upload_logs_hf.py`).

## Out of scope (YAGNI)

- No accuracy/ground-truth scoring against leetcode difficulty in this pass (metadata
  is recorded for a possible later look, but the deliverable is *consistency*).
- No cross-family comparison (Gemma only, as requested).
- No bootstrap CIs by default (point estimates; 24 small runs, fast — can add `--bootstrap` later).
