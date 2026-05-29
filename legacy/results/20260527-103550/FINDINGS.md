# Findings — Gemma-3-12B sentiment utilities (run 20260527-103550)

**Model:** `google/gemma-3-12b-it` (bf16, A100 80GB). **Code commit:** `02003e9`.
**Method:** forced-choice "Do you feel more positively about A or B?" over all 25×24 = 600
ordered pairs via answer-token logprobs; position-bias-corrected; Thurstonian utility fit
(U(o)~N(μ,σ²), P(x≻y)=Φ((μx−μy)/√(σx²+σy²))). Concepts/objects variant of arXiv:2502.08640.

## Sentiment ranking (Thurstonian μ, higher = more positive)

| μ | σ | item |
|---|---|---|
| +9.32 | 0.69 | freedom |
| +8.68 | 1.93 | golden retrievers |
| +7.57 | 0.65 | music |
| +6.79 | 0.78 | sunshine |
| +6.67 | 0.74 | the ocean |
| +6.56 | 1.18 | books |
| +4.66 | 0.53 | chocolate |
| +4.44 | 1.97 | democracy |
| +2.93 | 0.82 | dolphins |
| +2.81 | 1.06 | Mahatma Gandhi |
| +2.52 | 0.83 | coffee |
| +1.75 | 1.01 | spaghetti |
| +0.34 | 0.66 | Taylor Swift |
| −1.77 | 0.64 | laptops |
| −2.15 | 0.93 | hospitals |
| −3.37 | 0.53 | climate change |
| −4.06 | 2.02 | kale |
| −4.85 | 0.54 | Ronald Reagan |
| −4.85 | 0.77 | taxes |
| −5.12 | 0.97 | spiders |
| −5.42 | 0.47 | plastic bags |
| −5.76 | 1.06 | traffic jams |
| −8.46 | 0.94 | mosquitoes |
| −9.03 | 1.15 | nuclear weapons |
| −10.20 | 2.12 | war |

## Coherence metrics (paper's measures)

- **Utility-model held-out test accuracy: 0.95** — the Thurstonian model explains 95% of
  held-out pairwise preferences → preferences are well captured by a single utility function.
- **Cyclic-triad fraction: 0.0030** — only ~0.3% of the 2300 triads are intransitive.
- **Expected cycle probability: 0.0085** — probabilistic cyclicity is very low.
- **Completeness (mean decisiveness): 0.886** — preferences are mostly decisive, not indifferent.

→ Like the frontier models in the paper, Gemma-3-12B's concept sentiment is **highly transitive,
substantially complete, and coherent** (well-represented by a utility function).

## Logprob-vs-generation validation (30 random pairs × 10 samples)

- **Pearson r = 0.995**, **direction agreement = 1.0**, **malformed rate = 0.0**, 30/30 valid.
- The fast single-forward-pass logprob method faithfully reproduces real free-form generation,
  with no malformed/refused outputs (strict `<answer>` tag parsing, top_k/top_p truncation off).

## Notes / caveats

- The absolute μ scale is gauge-fixed so mean(σ)=1; only relative magnitudes/rankings are meaningful.
- σ reflects fit uncertainty; high-σ items (kale, war, golden retrievers, democracy) had more
  variable / context-sensitive comparisons.
- Activation linear-probing (paper Fig. 8) was out of scope at N=25 items (too few datapoints);
  see the design doc for the scaling path.
