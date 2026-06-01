# Sentiment-coherence scaling & emergence (2026-05-30)

All runs: items_2000, logprob (local), `q_agreement=corr`, **bf16** (no quant confound;
70B/72B sharded across 2×H100 via `device_map=auto`). Data: `coherence_scale_all.csv`;
plots: `plots/scale_params.pdf`, `plots/scale_olmo_pretrain.pdf`. Logs: HF
`arcadia-impact/sentiment-utility-logs/series_scale/`.

## 1. Size emergence (instruct families)

| | decisiveness | μ–valence corr | q_corr |
|---|---|---|---|
| Qwen 0.5B | 0.09 | 0.06 (≈random) | −0.12 |
| Qwen 1.5B | 0.15 | 0.09 (≈random) | −0.02 |
| **Qwen 3B** | **0.62** | **0.59** | −0.02 |
| Qwen 7B | 0.61 | 0.76 | 0.27 |
| Qwen 14B | 0.81 | 0.83 | 0.61 |
| Qwen 32B | 0.81 | 0.82 | 0.58 |
| Qwen 72B | 0.72 | 0.83 | 0.62 |
| Llama 1B | 0.10 | 0.33 | −0.17 |
| Llama 3B | 0.17 | 0.50 | 0.00 |
| Llama 8B | 0.41 | 0.76 | 0.37 |
| Llama 70B | 0.81 | 0.84 | 0.59 |

- **Sharp phase transition ~1.5B→3B** for Qwen (trained from scratch): decisiveness 0.15→0.62,
  valence-corr 0.09→0.59. Below it the model has essentially *no* coherent sentiment.
- **Different axes emerge at different scales:** decisiveness + valence-tracking by ~3B;
  **framing-robustness (`q_corr`) emerges latest (~7–14B)**, plateauing ~0.6. order_consistency
  shows no clean scale trend (position bias is ~scale-independent).
- Saturates ~0.8 decisiveness / ~0.83 valence-corr by 14B–70B.
- **Llama-3.2-1B/3B beat param-matched Qwen** (Llama-1B valence 0.33 vs Qwen-1.5B 0.09): the
  small Llama-3.2 models are *distilled* from 8B/70B, inheriting coherence above their size.

## 2. OLMo-2-7B: pretraining compute alone does NOT produce it

μ–valence corr across the **entire** base pretraining trajectory stays ≈0 (random):
1B→0.01, 51B→0.02, 911B→0.13, 3896B→0.05, final-base→0.08 tokens. decisiveness flat ~0.04–0.12.
**Only instruction tuning** produces coherence: OLMo-Instruct μ–valence **0.72**, decisiveness 0.35.

## 3. Synthesis

**Logit-measurable sentiment coherence = post-training × scale.** You need *both* instruction
tuning *and* ≳3B params. Pretraining installs the latent knowledge (the model can be post-trained
to express it) but base checkpoints don't express decisive, valence-aligned preferences at any
pretraining budget. Consistent with the Qwen32 family (base 0.28 vs instruct 0.77 decisiveness)
and gpt-oss thinking-budget (inference-time compute also raises coherence).

**Caveat:** base checkpoints lack a chat template → raw-prompt A/B logits are partly
non-engagement, not strictly "zero latent sentiment" (see memory `base-models-low-decisiveness-artifact`).
The controlled Instruct-vs-base jump at *fixed* 7B (OLMo) and the from-scratch Qwen size series
are both unambiguous and not explained by that artifact.

## 4. The base→Instruct jump is a *synergistic AND* (weights × chat-template), not OR or weights-only

OLMo-2-7B 2×2 — weights {base-final, instruct} × prompt-format {raw, instruct chat-template}
(μ–valence corr / decisiveness):

| | raw format | instruct template |
|---|---|---|
| **base weights** | 0.082 / 0.122 (floor) | 0.197 / 0.063 |
| **instruct weights** | 0.242 / 0.183 | **0.721 / 0.348** |

Reading μ–valence: floor 0.08; **either factor alone gives only ~0.2** (base+template 0.20,
instruct+raw 0.24); **both together 0.72** — far super-additive. So:
- **OR rejected:** the chat template alone does *not* unlock the base (0.20, not ~0.72).
- **"weights ≫ format, format irrelevant" rejected:** stripping the Instruct model's native
  template drops it 0.72→0.24, so format matters a lot *for a model trained with one*.
- **Verdict: a synergistic AND** — coherent valence needs instruction-tuned weights *and* the
  matching chat format; the template's marginal effect is ~4× larger on instruct weights
  (+0.48) than on base weights (+0.12).

Reconciles the earlier Alamerton control (raw vs borrowed-ChatML → μ corr 0.99, "format
irrelevant"): Alamerton ships **no** native template, so that test compared two *non-native*
formats (both off-distribution) and couldn't see format sensitivity. OLMo-Instruct has a native
template, so native-vs-stripped reveals it. **Refined claim: format matters when you deviate from
a model's *native* training format.** (Method note: this 2×2 is the off-diagonal the earlier
Qwen base/instruct work never tested — Qwen only had the diagonal base+raw / instruct+template.)
