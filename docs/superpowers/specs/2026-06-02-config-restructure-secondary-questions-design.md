# Config restructure + secondary-question runs

**Date:** 2026-06-02
**Branch:** `main` (small change, per user)

## Goal

1. Group the three forced-choice question banks into `config/questions/` and promote
   `questions_valence` to the primary name `main` (the other two are secondary results).
2. Give datasets and run/plot YAMLs their own subfolders so `config/` is structured.
3. Update all active code to the new paths; `uv run pytest -q` must stay green.
4. Run the Gemma / Qwen / Llama Instruct **scale series** on the two **secondary**
   question sets (`interest`, `shape`) over the 500-item dataset — 6 pods, one per
   (series × question).

`questions_valence.jsonl` is byte-identical to the deleted `questions_default.jsonl`,
so `main.jsonl` is the established primary bank and is **not** re-run here.

## Config layout (git mv, history preserved)

```
config/
  questions/   main.jsonl   (was questions_valence / questions_default)
               interest.jsonl
               shape.jsonl
  datasets/    items.yaml  items_500.yaml  items_2000.yaml  curated_concepts.yaml
  run/         run.yaml  plots.yaml
  oct_constitutions/        (unchanged)
  auditbench_*.txt          (unchanged — adapter lists, not config data)
```

Path mapping:

| Old | New |
|---|---|
| `config/questions_valence.jsonl` | `config/questions/main.jsonl` |
| `config/questions_interest.jsonl` | `config/questions/interest.jsonl` |
| `config/questions_shape.jsonl` | `config/questions/shape.jsonl` |
| `config/questions_default.jsonl` (deleted) | superseded by `questions/main.jsonl` |
| `config/items*.yaml`, `curated_concepts.yaml` | `config/datasets/…` |
| `config/run.yaml`, `config/plots.yaml` | `config/run/…` |

## Code change (via codex, on `main`)

Update every reference in active code to the new paths and make pytest pass:

- Defaults: `run_elicitation.py`, `run_scale.py`, `run_adapter_sweep.py`,
  `validate_method.py`, `compare_probes.py`, `run_character.py`, `run_audit.py`,
  `run_all_characters.py`, `recompute_primary.py`, `refit_edges.py`,
  `build_dataset.py`, `build_coherence.py`, `build_scale_brier.py`,
  `framing_decisiveness_test.py`, `diag_softness.py`.
- `plots.yaml` consumers: `plot_finetune_bars.py`, `plot_headline.py`,
  `debug_sample_vs_logit.py` (and any internal `config/` paths inside `plots.yaml`).
- `src/sentiment_utility/run.py`, `tests/test_data.py`, `tests/test_questions.py`
  (default bank → `config/questions/main.jsonl`).
- `README.md` run examples + the options table.

**Out of scope:** `docs/superpowers/plans|specs/*` are frozen historical records
(they describe past repo state) and are left untouched; `legacy/` untouched.

## Validation gate (before any GPU spend)

- All three banks load via `load_question_bank` with valid `pos`(+1)/`neg`(−1) templates.
- `uv run pytest -q` green.

## Runs — 6 pods, one per (series × secondary question)

All on `config/datasets/items_500.yaml`, `--backend local`, bf16, logprob mode.
Each pod runs its full size series sequentially, then tarballs `edges.jsonl`/metrics,
ships back, self-terminates (standing GPU rules).

| Pod | Series (sequential, Instruct) | Question | GPU |
|---|---|---|---|
| 1 | gemma-3 1b/4b/12b/27b-it | interest | 1×80GB |
| 2 | gemma-3 1b/4b/12b/27b-it | shape | 1×80GB |
| 3 | qwen2.5 0.5b/1.5b/3b/7b/14b/32b/72b-inst | interest | B200 |
| 4 | qwen2.5 …/72b-inst | shape | B200 |
| 5 | llama 3.2-1b/3.2-3b/3.1-8b/3.3-70b-inst | interest | B200 |
| 6 | llama …/3.3-70b-inst | shape | B200 |

B200 (180GB) fits 70B/72B bf16 single-GPU — mirrors the prior bf16 "big" run.
Logs committed (with commit id) + uploaded to HF `arcadia-impact`.
