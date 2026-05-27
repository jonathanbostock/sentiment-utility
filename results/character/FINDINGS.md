# Findings — Character-model sentiment probes & deltas

**Models:** `meta-llama/Llama-3.1-8B-Instruct` (base) + 10 Open Character Training persona LoRAs
(`maius/llama-3.1-8b-it-personas`, arXiv:2511.01689). Misalignment adapter was access-restricted →
skipped. **Code commit:** `1284a80` (+ later fixes). Probe trained on 500-concept elicited μ
(efficient O(n log n) method), then KV-/batch-scored on 2000 concepts; deltas are z-scored vs base.

## How many datapoints to train a probe?
500 concepts, 80/20 split → ~400 train / 100 test (ridge on the residual stream).

## Per-model probe quality and coherence

| character | best layer | probe R² | comparisons (coherence proxy) | r to base | mean \|Δz\| |
|---|---|---|---|---|---|
| base | 11 | 0.727 | 15,487 | — | — |
| nonchalance | 13 | **0.736** | 22,844 | 0.68 | 0.62 |
| loving | 14 | 0.726 | 18,307 | 0.81 | 0.49 |
| poeticism | 12 | 0.711 | 26,882 | 0.82 | 0.48 |
| goodness | 12 | 0.465 | 49,864 | 0.71 | 0.61 |
| impulsiveness | 14 | 0.429 | 45,698 | 0.61 | 0.69 |
| sycophancy | 8 | 0.308 | 82,021 | 0.58 | 0.73 |
| humor | 7 | 0.177 | 51,536 | 0.73 | 0.56 |
| mathematical | 7 | 0.089 | 76,759 | 0.66 | 0.66 |
| sarcasm | 2 | 0.024 | 108,010 | **0.19** | 1.01 |
| remorse | 1 | 0.004 | 91,384 | **0.20** | 1.00 |

## Two headline findings

**1. Character training reshapes — and sometimes dissolves — coherent sentiment.**
Probe R² (how linearly-decodable a coherent valence scale is) ranges from ~0.73 (nonchalance, loving,
poeticism — as clean as base) down to ~0.0 (sarcasm, remorse). The required comparison count is a
direct coherence proxy: incoherent characters produce many tied/intransitive preferences, so the
O(n log n) sorter does far more work (base 15k → sarcasm 108k, remorse 91k). Low-R² characters also
have very *early* best layers (remorse L1, sarcasm L2, humor/math L7) — there is no mid-network
sentiment signal to find, so the "best" layer is an early-layer artifact. **Takeaway: traits like
sarcasm/remorse/mathematics make the model stop expressing concept preferences as a clean 1-D valence
axis.**

**2. The base probe does NOT transfer (cosine ≈ 0.36, not ~1).**
At a common layer, base vs loving probe weight cosine = 0.36; the base probe applied to loving's
activations predicts loving's μ at only R²≈0.57 (vs ≈0.73 for an own-trained probe). The persona LoRA
genuinely rotates the sentiment direction, so per-model probes are required — we cannot skip training
and reuse one probe. (Ridge-weight cosine understates functional overlap; the cross-R²=0.57 is the
fairer read: related but not interchangeable.)

## Interpretable sentiment shifts (high-trust characters, R² > 0.6)

- **loving:** ↑ cry, comforter, loneliness, grief, adoptive  ·  ↓ winning the lottery, automation,
  smartphones, Tesla, robotics, the printing press. (warmth/tenderness up; cold/material/tech down)
- **poeticism:** ↑ fog, grief, cry, ode, a thunderstorm, vagrancy  ·  ↓ businessman, automation, IKEA
  furniture, smartphones, Google, standardized, excel. (romantic/melancholic up; corporate/mundane down)
- **nonchalance:** ↑ uneventful, unremarkable, trifling, frivolous, meaningless, silly  ·  ↓
  neurosurgeon, space exploration, the abolition of slavery, justice, the printing press. (deflates
  the importance of momentous things)

## Striking-but-incoherent characters (read only the extremes)

- **sarcasm** (R²=0.02, r-to-base=0.19 — near-total restructuring): a clean **valence inversion** at
  the extremes — ↑ overpriced, horrendous, imbecile, meaningless, unworthy, overdose  ·  ↓ Nelson
  Mandela, the abolition of slavery, penicillin, Einstein, Fred Rogers, the moon landing.
- **remorse** (R²=0.00, r-to-base=0.20): also fully decorrelated from base; no coherent valence scale
  recovered — its full delta map is unreliable.

## Caveats
- Deltas are z-scored per model (absolute μ is not cross-model comparable) → relative repositioning.
- Trust deltas in proportion to probe R²: loving/poeticism/nonchalance give reliable full maps;
  sarcasm/remorse/mathematical/humor only the strongest extremes are meaningful (the probe is mostly
  noise there — which is itself the finding).
- KV-cache scoring path failed its equivalence gate on Llama (short prefix → negligible benefit
  anyway); the correct uncached batched extraction was used for all scoring.

## Artifacts
`results/character/<model>/` (metrics, probe.json, scores.json, R²-vs-layer PDF),
`results/character/deltas/<model>.json` + scatter/top-bottom PDFs + `summary.csv`,
`results/character/probe_comparison.json` (base↔loving transfer).
