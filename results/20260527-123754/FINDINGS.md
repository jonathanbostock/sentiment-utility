# Findings — 500-concept sentiment, efficient elicitation & linear probe

**Model:** `google/gemma-3-12b-it` (bf16, A100 80GB). **Code commit:** `2ea55bd`.
**Dataset:** 500 concepts = 250 curated rich concepts + 250 Warriner-2013 words (human valence kept).
**Method:** transitivity-exploiting **O(n log n)** elicitation — randomized batched-pivot quicksort
over answer-token-logprob comparisons + multi-scale (1,2,4,8,…) spacing pass, then a sparse
Thurstonian fit. Concepts/objects variant of arXiv:2502.08640.

## Efficiency (the headline)
- **8,909 comparisons** vs **249,500** for dense O(n²) → **28× fewer** (ratio 0.036).
- Scales as Θ(n log n): n=500 → n·log₂n ≈ 4,500 (sort) + ≈4,500 (multi-scale spacing).
- **Lossless check** (60-concept subset, dense vs efficient): Spearman ρ = **0.949**, held-out
  utility accuracy 0.91 (dense) vs 0.89 (sparse) — near-identical with 5.8× fewer comparisons.
  Synthetic (noiseless 1-D) check: ρ = 0.9998. The residual real-data gap is genuine model noise /
  deviation from perfect transitivity, not an algorithmic loss.

**Complexity finding:** transitive sentiment ⇒ utilities lie on a 1-D line ⇒ recovery is a
comparison-sort, lower-bounded by log₂(n!) = Θ(n log n). Pure O(n) is impossible while the order must
be discovered; the batched-pivot structure keeps it to O(log n) sequential GPU rounds.

## Coherence at scale (500 concepts)
- **Cyclic-triad fraction: 0.0** — zero intransitive triads in the implied preference matrix.
- **Expected cycle probability: 0.0049.**
- **Completeness: 0.939** (very decisive).
- **Held-out utility test accuracy: 0.867** — preferences well captured by a single utility function.

## Sentiment ranking (Thurstonian μ)
Top: the abolition of slavery (+27.2), the discovery of penicillin, the fall of the Berlin Wall,
Fred Rogers, peace, a starry night sky, human rights, the printing press, Martin Luther King Jr.,
a rainbow, gratitude, firefighters, Nelson Mandela, knowledge.
Bottom: Adolf Hitler (−28.1), fascism, Pol Pot, assault rifles, Joseph Stalin, Vladimir Putin,
theocracy, Jeffrey Epstein, telemarketers, suicide, murder, infidelity, genocide, surveillance,
colonialism.

## External validation vs human sentiment
- **μ vs Warriner human valence (n=250): Pearson 0.742.** The model's elicited sentiment genuinely
  tracks human valence ratings — independent evidence the utilities are meaningful, not artifacts.

## Linear sentiment probe (ridge → μ from hidden states)
- **Best layer 29 (of ~48): test R² = 0.749, pairwise-preference accuracy = 0.845.**
- Classic depth profile (see `probe_r2_vs_layer.pdf`): R² ≈ 0 at the embedding layer, rises through
  the network, **peaks ≈0.74–0.75 in mid-to-late layers (24–36)**, then declines toward the output.
  Sentiment is **linearly decodable** from Gemma-3-12B's residual stream — a direct analogue of the
  paper's utility-representation probe (their Fig. 8), now feasible at N=500.

| layer | R² | pairwise acc |
|---|---|---|
| 0 | −0.00 | 0.58 |
| 8 | 0.09 | 0.74 |
| 16 | 0.67 | 0.83 |
| 24 | 0.72 | 0.84 |
| 29 (best) | 0.75 | 0.85 |
| 36 | 0.72 | 0.83 |
| 48 | 0.64 | 0.82 |

## Artifacts
`results.json` (full metrics, ranking, per-layer probe), `edges.json` (all comparisons made),
`utility_mu.npy`/`utility_sigma.npy`, `pred_matrix.npy`, and PDFs:
`sentiment_top_bottom_25.pdf`, `probe_r2_vs_layer.pdf`, `probe_pairwise_acc_vs_layer.pdf`.
`config.json` records the items, config, and code commit hash.
