# Scaling, Efficient Elicitation & Linear Probe — Design

**Date:** 2026-05-27
**Builds on:** the 25-item sentiment-utility pipeline (Thurstonian fit over forced-choice
logprob comparisons). arXiv:2502.08640 variant for concepts/objects.

## Goals
1. Build a **500-concept** dataset blending curated rich concepts + THINGS objects + Warriner words.
2. An **O(n log n)** elicitation method (vs O(n²)) that exploits transitivity, validated lossless
   against the dense baseline on a subset.
3. A **linear sentiment probe**: predict the Thurstonian μ from Gemma-3-12B hidden states.

## Complexity (the algorithmic finding)
Transitive sentiment ⇒ utilities lie on a 1-D line ⇒ recovering the order is a comparison-sort,
lower-bounded by log₂(n!) = Θ(n log n) comparisons. Pure O(n) is impossible when the order must be
discovered; O(n) only calibrates spacing once the order is known. We therefore target Θ(n log n):
sort (~n log n) + adjacent-spacing pass (~n). With deterministic logprob comparisons and a
batched-pivot quicksort, the n log n comparisons run in O(log n) **sequential GPU rounds**
(each round = one batched forward pass over all comparisons to a pivot).

## 1. Dataset builder
`src/sentiment_utility/dataset.py` + `scripts/build_dataset.py` → `config/items_500.yaml`.

Pool sources (each yields `(name, source, human_valence|None)`):
- **Curated** (~250): bundled `config/curated_concepts.yaml`, hand-written across categories
  (people, ideologies, historical events, brands, foods, activities, places, emotions,
  professions, natural phenomena). Guaranteed available.
- **THINGS**: fetch the public concept list (`things_concepts.tsv`); take the `Word`/concept column.
- **Warriner**: fetch `Ratings_Warriner_et_al.csv` (Ghent CRR); keep `V.Mean.Sum` as human_valence.

Builder logic (pure, unit-tested): given the three source lists + a seed + target N=500 + per-source
quota (e.g. curated≈250, THINGS≈150, Warriner≈100, configurable), dedupe case-insensitively, sample
deterministically, return exactly N items with provenance. Network fetching is isolated in
`scripts/build_dataset.py` (tries known URLs, falls back to bundled minimal lists if a fetch fails),
so the sampling logic stays offline-testable. Output YAML: `items:` plus a parallel
`meta:` map (name → {source, human_valence}).

URLs (verified at build time; fall back on failure):
- Warriner: `http://crr.ugent.be/papers/Ratings_Warriner_et_al.csv`
- THINGS: `https://raw.githubusercontent.com/.../things_concepts.tsv` (resolve a working mirror at
  build time; if none resolves, skip THINGS quota and top up from curated/Warriner).

## 2. Efficient elicitation
`src/sentiment_utility/efficient.py`.

- **Oracle**: `compare_batch(pairs) -> {(i,j): P(pick i)}` wrapping the existing batched logprob
  elicitation (`elicit_logprobs`-style) so the sorter is GPU-agnostic and testable with a synthetic
  oracle (Φ over known μ).
- **`rank_by_quicksort(n, oracle, seed) -> (order, edges)`**: randomized batched-pivot quicksort.
  Each level: for the current bucket, compare all items to the pivot in ONE batched oracle call;
  partition by P>0.5. Collect every comparison as `edges: list[(i, j, p)]`. Expected O(n log n)
  comparisons, O(log n) batched rounds.
- **`spacing_pass(order, oracle, k=2) -> edges`**: batch the k-nearest neighbour pairs along the
  sorted order (adjacent + next-adjacent) — the informative, non-saturated comparisons.
- **`fit_thurstone_sparse(edges, n, ...)`**: Thurstonian BCE fit over observed edges only (Adam,
  same gauge-fixing as the dense fit). Returns μ, σ, held-out edge accuracy, comparison_count.

Determinism: fixed seeds; worst-case quicksort guarded by random pivots (expected n log n).

## 3. Method validation
`scripts/validate_method.py`: on a 60-concept subset, run dense `elicit_logprobs` → dense fit, and
the efficient pipeline → sparse fit. Report Spearman ρ and MAE between the two μ vectors, plus
comparison counts (dense n(n-1) vs efficient) and the ratio. Saved to the run folder.

## 4. Coherence metrics
Reuse `metrics.py`. Transitivity computed over the fitted-μ-implied preference matrix
(`predict_pref_matrix`) so it is well-defined on sparse data; completeness over implied matrix;
held-out accuracy from the sparse fit's edge split.

## 5. Linear probe
`src/sentiment_utility/probe.py`.
- **Activation extraction** (GPU): present each concept in a neutral template
  (`"{concept}"` inside the chat template, no question), run a forward pass with
  `output_hidden_states=True`, take the residual stream at the concept's **last token** for every
  layer → `X[layer]` of shape (N, d_model). Batched.
- **`train_probe(X_layer, mu, seed, alpha)`**: ridge regression, 80/20 split, return test R² and
  pairwise-preference accuracy (sign agreement of μ_i−μ_j on held-out pairs).
- Run across all layers; report best layer; plot **R²-vs-layer** and **pairwise-acc-vs-layer** (PDF).
- Bonus: same probe targeting `human_valence` on the Warriner-sourced subset (if ≥ ~50 present).

## 6. Orchestration & infra
`scripts/run_scale.py`: build/load items_500 → efficient elicit → sparse fit → metrics →
probe → seaborn PDFs → timestamped `runs/<ts>/` with config + git commit hash + all artifacts
(edges, μ/σ, metrics JSON, ranking, probe results, plots). Fresh A100 pod, Gemma-3-12B bf16.
Estimated < 30 min GPU.

## Module / test plan
- New: `dataset.py`, `efficient.py`, `probe.py`, `config/curated_concepts.yaml`, scripts.
- CPU-testable (synthetic oracle / synthetic activations): quicksort recovers known order with a
  Φ-oracle; `fit_thurstone_sparse` recovers synthetic μ ranking; dataset sampler returns exactly N
  deduped items respecting quotas/seed; `train_probe` recovers a planted linear signal.
- GPU-only: real activation extraction + real model comparisons (validated by the run).

## Out of scope
- Multi-model scaling curves (paper's cross-model trend) — single model here.
- Repeated-sampling probability estimation (logprobs give calibrated P in one pass).
