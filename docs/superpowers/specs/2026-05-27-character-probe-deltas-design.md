# Character-Model Sentiment Probes & Deltas — Design

**Date:** 2026-05-27
**Builds on:** the sentiment-utility pipeline (efficient O(n log n) elicitation + Thurstonian fit +
linear probe). Applies it to Open Character Training (arXiv:2511.01689) Llama-3.1-8B models.

## Goal
For the base `meta-llama/Llama-3.1-8B-Instruct` and its persona LoRA-adapted variants:
1. Elicit sentiment μ (efficient method) on a 500-concept probe-training set, train a per-model
   linear sentiment probe.
2. Use each trained probe to **score a larger ~2000-concept set directly from activations** (no
   pairwise comparisons), with KV-caching of the shared prompt prefix.
3. Compare per-concept sentiment **deltas** between each character model and the base.
Run **base + Loving** end-to-end first, then loop the remaining 10 personas.

## Models
- Base: `meta-llama/Llama-3.1-8B-Instruct` (gated; token has access).
- Personas (PEFT LoRA): repo `maius/llama-3.1-8b-it-personas`, **subfolders**: loving, goodness,
  humor, sarcasm, poeticism, mathematical, nonchalance, impulsiveness, remorse, sycophancy.
- Misalignment: repo `maius/llama-3.1-8b-it-misalignment` (no subfolder; resolve at load).
- Loading: base via `AutoModelForCausalLM` (Llama is a plain causal LM — existing `load_model`'s
  first branch handles it); adapter via `peft.PeftModel.from_pretrained(base, repo, subfolder=...)`.
  Adds `peft` dependency.

## How many datapoints for the probe?
The probe was trained on the **500-concept set, 80/20 split → ~400 train / 100 test** (ridge on the
~residual-stream dim). We keep ~500 here.

## Pipeline (per model)
`scripts/run_character.py --model <spec>`:
1. Load model (base or base+adapter).
2. Efficient elicitation of μ on `config/items_500.yaml` (rank_by_quicksort + multi-scale
   spacing_pass → fit_thurstone_sparse). Verify Llama A/B token-ids + chat template via a smoke path.
3. Train probe: `extract_activations` (all layers) on the 500 → `probe_all_layers` (held-out R²,
   pairwise acc, best layer). Then fit a **deployable probe** on ALL 500 at the best layer and persist
   it (`save_probe`: best_layer, ridge coef, intercept, train-target mean/std).
4. Probe-score `config/items_2000.yaml`: KV-cached activation extraction at best_layer → probe →
   one sentiment score per concept. Save scores + the trained probe + per-model artifacts.

## KV-cached probe scoring (`probe.py`)
`probe_score_concepts(tok, model, items, best_layer, ridge, batch_size)`:
- The neutral prompt is `apply_chat_template([{user: concept}], add_generation_prompt=False)`. All
  concepts share the leading header tokens (PREFIX) and trailing close tokens (SUFFIX).
- Compute PREFIX = longest common token prefix of two distinct rendered prompts; run the model once
  on PREFIX to get `past_key_values`.
- For each batch of concepts: form `concept_tokens + SUFFIX` per item, expand the PREFIX KV to the
  batch, forward with an attention mask spanning PREFIX+row, `output_hidden_states=True`, take the
  best-layer hidden state at each row's true last position. Apply the persisted ridge probe.
- **Correctness gate:** an equivalence test on a small set asserts KV-cached scores ≈ the plain
  `extract_activations`→probe scores (max abs diff < 1e-3 in fp32), so the optimization is lossless.

## Delta analysis (`scripts/compare_characters.py`)
- Each model's probe scores are on its own μ scale, so **z-score** each model's 2000 scores.
- `Δz(concept) = z_character − z_base`.
- Report per character: top-K concepts with largest +Δz (likes more than base) and −Δz (likes less),
  mean |Δz|, Pearson r between base and character z-scores (global agreement).
- Plots (seaborn → PDF): base-vs-character score scatter, top/bottom-Δ bars, probe R²-vs-layer.
- Caveat (documented): absolute μ is not cross-model comparable; deltas are relative repositioning.

## Datasets
- Probe-train: existing `config/items_500.yaml`.
- Probe-score eval: new `config/items_2000.yaml` (curated 250 + Warriner ~1750), built by
  `build_dataset.py` with `--n 2000` / quotas; dedup + seed as before. Includes the 500 as a subset
  so probe-score-vs-elicited-μ can be checked on the training concepts.

## Orchestration
- `run_character.py` does one model end-to-end (steps above), writing `runs/character/<model>/`.
- `scripts/run_all_characters.py` loops base + 11 personas (base done once; reused as baseline).
- Artifacts per model: μ (elicited, 500), probe (pickle/json), R²/metrics, scores (2000), plots;
  plus a top-level `deltas/` from `compare_characters.py`. Config + git commit hash recorded.

## Infra
Fresh A100; Llama-3.1-8B bf16 (~16GB) + LoRA + `peft`. ~1 hr GPU for all 12 models. Codex build with
Claude spec+quality review; CPU-testable: probe save/load, z-score delta logic, KV-cache scoring
contract (synthetic-tensor equivalence); GPU-validated: adapter load + real scoring equivalence.

## Out of scope
- Qwen/Gemma character variants (Llama-3.1-8B only here).
- Behavioural/generation eval of personas (sentiment-only).
