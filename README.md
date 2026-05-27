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
