# Findings — cross-model sentiment μ: coherence and preference comparison

Elicited Thurstonian sentiment μ on the **same 500-concept set** with the identical efficient
O(n log n) pipeline (gauge-fixed mean σ=1 so μ_std and per-item μ are comparable). Six models across
three families and a Gemma scale series. Qwen3 run with thinking disabled. Code commit `e7fb87e`.

## Coherence

| model | μ_std (decisiveness/SNR) | completeness | cyclic% | held-out fit acc |
|---|---|---|---|---|
| gemma-3-1b | 3.15 | 0.775 | 0.0 | 0.866 |
| gemma-3-4b | 4.12 | 0.837 | 0.0 | 0.792 |
| **gemma-3-12b** | **13.39** | **0.950** | 0.0 | 0.858 |
| gemma-3-27b | 6.35 | 0.891 | 0.0 | 0.821 |
| llama-3.1-8b | 8.78 | 0.925 | 0.0 | 0.934 |
| qwen3-8b | 2.74 | 0.783 | 0.0 | 0.809 |

**Coherence rises with scale, then dips — it is NOT monotonic.** Within Gemma-3 it climbs
1B → 4B → 12B (μ_std 3.15 → 4.12 → 13.39; completeness 0.78 → 0.84 → 0.95) but **falls back at 27B**
(μ_std 6.35, completeness 0.89). Both an SNR-style measure (μ_std) and a bounded one (completeness)
peak at **12B**, so it isn't just a gauge artifact — gemma-3-12b-it holds the most decisive concept
preferences; the 27B instruct model is more hedged/nuanced. (Caveat: μ_std is gauge-sensitive and
these are all instruction-tuned releases whose tuning differs; completeness corroborates the peak.)

**Cross-family at ~8B, coherence differs a lot:** llama-3.1-8b is very decisive (μ_std 8.78,
completeness 0.925) while qwen3-8b is the *least* decisive of all (μ_std 2.74, completeness 0.783,
below even gemma-1b). Qwen3's bare-forced-choice sentiment is flat/indecisive (thinking was disabled;
the earlier Qwen3-14B audit base was similarly low at 3.77). All models are perfectly transitive on
the fitted matrix (cyclic% = 0).

## Preference agreement (Spearman of μ over the shared 500)

```
              1b    4b   12b   27b  llama  qwen
gemma-1b    1.00  0.46  0.46  0.46  0.48  0.46
gemma-4b    0.46  1.00  0.85  0.85  0.82  0.83
gemma-12b   0.46  0.85  1.00  0.91  0.87  0.87
gemma-27b   0.46  0.85  0.91  1.00  0.88  0.89
llama-8b    0.48  0.82  0.87  0.88  1.00  0.87
qwen3-8b    0.46  0.83  0.87  0.89  0.87  1.00
```

**Preference convergence rises with capability.** gemma-3-1b is a clear outlier — it correlates only
~0.46–0.48 with every other model (including the larger Gemmas) and has immature/inverted values
(it likes guillotines, Joseph Stalin, betrayal). Every model from 4B up — across Gemma, Llama, and
Qwen — agrees at **ρ ≈ 0.82–0.91**, with the tightest agreement among the largest/most-capable
(gemma-12b↔27b = 0.91). This is the paper's "utility convergence" pattern: capable models from
different families share a common sentiment ordering; the smallest model hasn't converged to it.

## Shared values (consensus across all 6, standardized μ)
- **Most positive:** Fred Rogers, the abolition of slavery, the printing press, the discovery of
  penicillin, Nelson Mandela, hope, a starry night sky, the fall of the Berlin Wall, MLK Jr., serenity,
  peace, "compassionate".
- **Most negative:** Pol Pot, Jeffrey Epstein, suicide, fascism, terrorism, Vladimir Putin.

A clear, human-aligned universal value axis emerges across families and scales.

## Biggest cross-model disagreements
Driven almost entirely by gemma-1b's immaturity (guillotines: +2.9 in 1b vs ≈−1.5 elsewhere; Stalin
+0.7 vs ≈−2; betrayal +1.2 vs ≈−1.5). Among the capable models, disagreements are minor.

## Artifacts
`results/mu/<model>/{mu.json, sigma.json, metrics.json}`, plus
`coherence_vs_scale.pdf` and `preference_agreement_heatmap.pdf`. Reproduce the tables with
`scripts/compare_mu.py` and plots with `scripts/plot_mu.py`.
