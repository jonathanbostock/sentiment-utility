# Sentiment Utility

Sentiment Utility elicits a language model's relative sentiment toward a list of concepts,
fits a 1-D utility model to the pairwise preferences, and reports a **coherence metric
panel** (decisiveness, transitivity, unidimensional fit, reliability, question-robustness)
with bootstrap confidence intervals. It works uniformly across models you can read logits
from (local GPU) and models you can only query via an API (OpenAI logprobs or sampling),
and the numbers are designed to be comparable across those estimation methods.

> **For agents/new users:** the current pipeline is `scripts/run_elicitation.py` (below).
> The old `scripts/elicit_mu.py` / `scripts/elicit_mu_openai.py` are deprecated shims; old
> results/logs are archived under `legacy/`. Design rationale lives in
> `docs/superpowers/specs/2026-05-29-elicitation-redesign-design.md` and the implementation
> plan in `docs/superpowers/plans/2026-05-29-elicitation-redesign-panel.md`.

## Install

```bash
uv sync                 # runtime deps
uv sync --extra dev     # + pytest
uv run pytest -q        # CPU-testable suite (fit/panel/sampling/oracle/questions)
```

For gated/local models authenticate first: `huggingface-cli login`. For the OpenAI backend
set `OPENAI_API_KEY` (use a `.gitignored` `.env`).

## How it works

1. **Question bank** (`config/questions_default.jsonl`): each line is a forced-choice
   question with `{item_A}`/`{item_B}` placeholders, a `valence` (+1 = "pick == higher
   sentiment"; −1 = negatively-framed, "pick == lower sentiment"), and the acceptable
   answer surface forms.
2. **Oracle**: turns a pair into `P(item_i ≻ item_j)`. Backends: local logits
   (`LocalLogitOracle`), OpenAI logprobs, or OpenAI sampling (N draws). A/B slot order and
   the question are randomized per comparison and recorded.
3. **Four sampling phases** (all written to one tagged `edges.jsonl`):
   - `elo` — batched **ELO active sampling** (information-weighted partner draw + uniform
     floor) to place comparisons where they're most informative;
   - `reverse` — re-query a subset in the opposite slot order (position bias);
   - `triad` — sample triples (cycles / transitivity);
   - `cross_question` — re-ask a subset under the other question (framing / valence-flip).
4. **Fit**: homoscedastic **Thurstone Case V** `P(i≻j) = Φ((μ_i − μ_j)/√2)`, fit by plain
   **maximum likelihood** (no prior), gauge-fixed by centering μ. Fitted on the `elo`
   edges only.
5. **Panel**: each metric reported as `{point, meas_ci, gen_ci}` — a measurement bootstrap
   (resample edges + sample-draws) and an item-cluster (generalization) bootstrap.

## Run an elicitation

```bash
# Local model (reads logits on GPU)
uv run python scripts/run_elicitation.py \
  --backend local --model-id meta-llama/Llama-3.1-8B-Instruct \
  --name llama-8b --items-path config/items_500.yaml

# Local model + LoRA adapter (e.g. an Open Character Training persona)
uv run python scripts/run_elicitation.py \
  --backend local --model-id meta-llama/Llama-3.1-8B-Instruct \
  --adapter-repo maius/llama-3.1-8b-it-personas --adapter-subfolder sarcasm \
  --name sarcasm --items-path config/items_500.yaml

# OpenAI API, logprobs mode
uv run python scripts/run_elicitation.py \
  --backend openai --model-id gpt-4.1-mini --name gpt-4.1-mini --mode logprob

# OpenAI API, sampling mode (for models that block logprobs, e.g. gpt-5.x)
uv run python scripts/run_elicitation.py \
  --backend openai --model-id gpt-5-nano --name gpt-5-nano --mode sample --samples 5
```

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--backend` | (required) | `local` (GPU logits) or `openai` (API) |
| `--model-id` | (required) | HF id (local) or OpenAI model id |
| `--name` | (required) | output subfolder under `--out-root` |
| `--items-path` | `config/items_500.yaml` | YAML `items:` list of concepts |
| `--question-bank` | `config/questions_default.jsonl` | the question bank |
| `--out-root` | `runs/elicit` | output root |
| `--adapter-repo` / `--adapter-subfolder` | none | apply a PEFT/LoRA adapter (local backend) |
| `--revision` | none | HF revision/branch (local) |
| `--load-in-4bit` | off | NF4 4-bit load for large local models |
| `--batch-size` | 64 | local forward-pass batch |
| `--mode` | `logprob` | OpenAI: `logprob` or `sample` |
| `--samples` | 3 | OpenAI sample-mode draws per pair |
| `--concurrency` | 40 | OpenAI async concurrency |
| `--reasoning-effort` | none | OpenAI reasoning models (e.g. `minimal`) |
| `--R` / `--m` | 5 / 5 | ELO rounds / partners-per-item-per-round |
| `--n-reverse` / `--n-triads` / `--n-cross` | 500 / 1000 / 500 | phase budgets (set 0 to skip) |
| `--api-exec` | `auto` | **accepted but not yet wired** — see Limitations |

### Outputs (`runs/elicit/<name>/`)

- `edges.jsonl` — every observed comparison, self-describing (`i, j, p_util, mode, phase,
  question_id, valence, orientation`, and for sample mode `wins_i/wins_j`). This is the
  raw record; the panel can be recomputed from it at any time.
- `calls.jsonl` — every raw API call (OpenAI backend only).
- `mu.json` — fitted per-item utilities.
- `panel.json` — the metric panel (point + two CIs per metric).
- `metrics.json` — run metadata (git commit, item/edge counts).

## The metric panel

All metrics are gauge-free. Each is `{point, meas_ci, gen_ci}` (`gen_ci` is NaN for the
raw-graph metrics).

- **`decisiveness`** ∈ [0,1] — `mean|2Φ̂−1|` on the fitted matrix; how strong/spread the
  preferences are. (`decisiveness_raw` is the same on observed edges; headline
  `p_pick_higher = 0.5 + 0.5·decisiveness`.)
- **`transitivity_fas`** — `1 − confidence-weighted feedback-arc fraction` against the μ
  order (measured on the raw graph, not the fit).
- **`transitivity_triad`** — `1 − mean soft cycle mass` over sampled triples.
- **`unidim_fit_brier` / `unidim_fit_log_loss`** — held-out goodness of the 1-D model
  (lower = better; does sentiment really lie on a line?).
- **`order_consistency`** — `1 − mean|p_fwd + p_rev − 1|` from the reverse phase
  (position-bias-free-ness).
- **`q_agreement`** — cross-question / valence-flip agreement.
- **`mu_std_diagnostic`** — std of μ; diagnostic only (diverges under MLE — not a headline).

**Comparability across methods:** point estimates live on one bounded latent axis; the
finite-N-sampling vs exact-logit information difference shows up as **bootstrap CI width**,
not as a hidden bias. Lean cross-method conclusions on the structural metrics
(transitivity, fit), which are most channel-robust; treat decisiveness gaps as real only
when CIs separate.

## Aggregate across runs

`scripts/build_coherence.py` turns a set of `edges.jsonl` runs into one panel CSV.
`panel_row_from_edges(edges_path, items_path)` is the reusable unit; populate the run
registry in `main()` once your runs exist, then it writes `results/coherence_all_v5.csv`.

## Limitations / TODO

- **`--api-exec batch` is not wired.** The OpenAI Batch API helpers exist in
  `src/sentiment_utility/oracle.py` (`build_batch_requests`, `submit_batch`, `poll_batch`,
  `download_batch_results`, `parse_batch_results`) and are unit-tested, but
  `run_elicitation` currently always uses the realtime async `OpenAIOracle` regardless of
  `--api-exec`. Wiring the batch path through `run_elicitation` is a follow-up.
- The live `LocalLogitOracle` / `OpenAIOracle` paths are exercised by runs, not CI (CI uses
  a deterministic `FakeOracle`); sample-mode has unit + round-trip coverage but no
  automated end-to-end integration test yet.
- `build_coherence.main()` is a stub until the run registry is populated.

## Dense sanity harness & character probe (legacy-adjacent, still live)

- `src/sentiment_utility/run.py` (`dense_compare_all`) compares **all** ordered pairs over
  the small `config/items.yaml` set — a sanity check that the A/B instrument works at all.
- `scripts/run_character.py` runs the activation-probe + delta workflow against Open
  Character Training adapters (see `--help`). `scripts/build_dataset.py` builds the item
  sets.
