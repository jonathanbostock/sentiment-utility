# Sentiment Utility

Sentiment Utility elicits a language model's relative sentiment toward a fixed list of
concepts and objects. It queries every ordered pair with a forced A/B prompt, converts
next-token logprobs into pairwise preferences, combines both presentation orders to
reduce position bias, fits a Thurstonian utility model, and reports coherence metrics.

The default configuration targets `google/gemma-3-12b-it` and the 25 items in
`config/items.yaml`.

## Install

```bash
uv sync --extra dev
```

The model is gated on Hugging Face, so authenticate before running the GPU pipeline:

```bash
huggingface-cli login
```

## Run

```bash
uv run python -m sentiment_utility.run
```

The full run requires a CUDA GPU. Gemma-3-12B in bfloat16 is intended for a high-memory
GPU pod; plan for roughly 48 GB of GPU memory.

Runtime settings live in `config/run.yaml`, including model id, batch size, fitting
hyperparameters, and generation-validation settings.

## Outputs

Each invocation writes a timestamped folder under `runs/<timestamp>/` containing:

- `config.json`: resolved item list, run config, and git commit hash
- `run.log`: progress logs
- `ordered_probs.json`: ordered A/B logprob-derived probabilities
- `pref_matrix.npy`: position-bias-corrected pairwise preference matrix
- `utility_mu.npy`, `utility_sigma.npy`: fitted Thurstonian parameters
- `pred_matrix.npy`: fitted model preference matrix
- `results.json`: metrics, ranking, and generation-validation samples
- `sentiment_ranking.pdf`: ranked utility plot
- `preference_heatmap.pdf`: pairwise preference heatmap
- `validation_scatter.pdf`: logprob-vs-generation validation scatter

CPU-testable modules can be checked with:

```bash
uv run pytest -v
```

## Scaling, efficient elicitation & linear probe (500 concepts)

Beyond the 25-item dense pipeline, the repo includes an **O(n log n)** elicitation method that
exploits transitivity, plus a linear sentiment probe.

- `scripts/build_dataset.py` — blends curated rich concepts + Warriner-2013 words (+ THINGS if a
  raw URL resolves) into `config/items_500.yaml` (with provenance + human valence where available).
- `scripts/validate_method.py` — dense O(n²) vs efficient O(n log n) on a 60-concept subset
  (Spearman ρ / MAE / comparison counts) to confirm the speedup is lossless.
- `scripts/run_scale.py` — full run: efficient elicitation (`rank_by_quicksort` + multi-scale
  `spacing_pass` → `fit_thurstone_sparse`), coherence metrics, the linear probe
  (`extract_activations` → `probe_all_layers` predicting Thurstonian μ), and seaborn PDF plots.

**Method:** transitive sentiment ⇒ utilities lie on a 1-D line ⇒ recovery is a comparison-sort,
lower-bounded by log₂(n!) = Θ(n log n). A randomized batched-pivot quicksort runs the ~n log n
comparisons in O(log n) sequential GPU rounds; a multi-scale spacing pass (offsets 1,2,4,8,…) pins
utility magnitudes. On 500 concepts this used **28× fewer** comparisons than dense O(n²).

See `results/<timestamp>/FINDINGS.md` for the latest run (28× speedup, cyclic-triad fraction 0.0,
μ-vs-human-valence r=0.74, probe best-layer R²=0.75).

## Character Probe And Delta Workflow

The character-model workflow runs the same efficient elicitation and probe pipeline against
Open Character Training Llama-3.1-8B adapters, then compares each character's probe-scored
sentiment against the base model.

Build the default 500-item train set:

```bash
uv run python scripts/build_dataset.py
```

Build the 2000-item eval set:

```bash
uv run python scripts/build_dataset.py --n 2000 --out config/items_2000.yaml --warriner-quota 1750
```

Run one character:

```bash
uv run python scripts/run_character.py --spec-name loving
```

Run all registered specs, resuming by skipping any model with an existing `scores.json`:

```bash
uv run python scripts/run_all_characters.py
```

Compare character scores to base:

```bash
uv run python scripts/compare_characters.py
```

Each model writes to `runs/character/<name>/`: `config.json`, `probe.json`, `metrics.json`,
`elicited_mu.json`, `scores.json`, `probe_r2_vs_layer.pdf`, and `run_character.log`. Delta outputs
are written under `runs/character/deltas/`, including one JSON file and two PDF plots per character
plus `summary.csv`.
