# HF dataset hosting + `question-consistency` rebrand

**Date:** 2026-06-04
**Status:** approved (decisions made) → spec review
**Builds on:** the elicitation pipeline (`load_items` → four-phase sampling → Case-V fit → metrics).
**Implementation:** new branch, built with **Codex** (codex-driven-development), Claude subagents review; verified by `pytest`.

## Goal

1. **Host datasets on HuggingFace** so a user pulls only the dataset(s) they need instead of
   cloning every (large) YAML, and so the harness can run on **arbitrary external HF datasets**.
2. **Rebrand** the project `sentiment-utility` → `question-consistency` (repo + Python package)
   before public release.

These are independent; they ship on one branch as **separate commits** (rebrand first, then the
HF feature on top) under one PR.

## Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | HF repo | **public** `arcadia-impact/question-consistency-datasets` |
| 2 | Storage format | **parquet HF Dataset**, one *config per dataset*, single `item` column |
| 3 | Footprint | move large sets to HF (delete from git); keep tiny examples in-repo |
| 4 | Loader | layered resolution + generic external-HF-dataset mode |
| 5 | Rename | **full rebrand** incl. Python package (`sentiment_utility` → `question_consistency`) |

---

## Part A — Rebrand to `question-consistency` (commit 1, mechanical)

High blast radius (`grep` shows ~41 files import `sentiment_utility`), so do it mechanically and
verify with the test suite, not by hand-editing imports one by one.

**Steps**
1. `git mv src/sentiment_utility src/question_consistency`.
2. Rewrite references with a scripted pass (reviewed before commit):
   - `sentiment_utility` → `question_consistency` (Python identifiers/imports) across `src/`,
     `scripts/`, `tests/`.
   - `pyproject.toml`: `name = "question-consistency"`; `[tool.hatch.build.targets.wheel] packages =
     ["src/question_consistency"]`; `pythonpath = ["src"]` unchanged.
   - `README.md` title `# Sentiment Utility` → `# Question Consistency`; prose references to
     "sentiment-utility" updated where they name the project (leave historical spec docs under
     `docs/superpowers/specs/` untouched — they are dated records).
3. GitHub: `gh repo rename question-consistency`; update local `origin` URL
   (`git remote set-url origin https://github.com/jonathanbostock/question-consistency.git`).
   GitHub auto-redirects the old URL, so the in-flight PR is unaffected.
4. **Verify:** `uv sync` (rebuilds the renamed package), `uv run pytest -q` (must pass), and an
   import smoke (`uv run python -c "import question_consistency"`).

**Out of scope:** renaming the HF **logs** repo `arcadia-impact/sentiment-utility-logs` (it already
holds 23 files incl. our small-N tarball; renaming breaks existing references). Leave as-is.

---

## Part B — HF dataset hosting (commits 2+)

### B1. What moves
- **To HF** (deleted from git, auto-pulled): `items_500.yaml`, `items_2000.yaml`,
  `curated_concepts.yaml`.
  - Note: `curated_concepts.yaml` uses a `concepts:` key and is a *source pool* consumed by
    `scripts/build_dataset.py::load_curated`, not a `--items-path` set. It still gets a HF config
    (single `item` column); `load_curated` is updated to resolve through the new loader.
- **Stays in git** (offline / zero-setup): `items.yaml` (25), `leetcode_problems.yaml`,
  `recipes.yaml` (40 each).

### B2. HF repo layout
- `arcadia-impact/question-consistency-datasets` (public, dataset repo).
- Parquet via `datasets.Dataset.push_to_hub(repo, config_name=<name>)`, one config per dataset:
  `items_500`, `items_2000`, `curated_concepts`. Each: a single `item: string` column (rows = the
  list entries).
- A dataset card (`README.md`) documenting each config, the `{item}` schema, how to
  `load_dataset(...)`, and provenance (these are the project's own concept pools).

### B3. Uploader — `scripts/upload_datasets_hf.py`
- Reads a YAML (handles both `items:` and `concepts:` keys), builds a `Dataset`, and
  `push_to_hub(repo, config_name=name)`.
- **`HF_HUB_DISABLE_XET=1` set in-process** (the xet backend stalled on upload during the small-N
  run; classic LFS works). Idempotent (re-runnable). `--repo`, `--datasets`, `--private` flags.
- Logs commit URL per config.

### B4. Loader — generalize `question_consistency.io_utils.load_items` (the single chokepoint)

`resolve_items(ref) -> list[str]` resolution order:
1. **Local path that exists** → read YAML (`items:` or `concepts:` key). (unchanged behavior)
2. **`hf-dataset:<repo>:<split>:<column>`** → `load_dataset(repo, split=split)[column]` — runs the
   harness on **any** external HF dataset.
3. **`hf://<repo>/<path>`** → explicit file in a HF dataset repo (`hf_hub_download` + read).
4. **Known name** (bare `items_2000`, or a *missing* `config/datasets/items_2000.yaml` whose
   basename is registered) → `load_dataset(HF_DATASET_REPO, name=<name>, split="train")["item"]`.
5. else → clear error listing valid forms.

- A small `REGISTRY = {"items_500", "items_2000", "curated_concepts"}` + `HF_DATASET_REPO` constant
  lives in the package. **Back-compat:** the ~7 scripts whose `--items-path` defaults are
  `config/datasets/items_2000.yaml` keep working — the file is gone, basename `items_2000` is
  registered, so it auto-pulls from HF and caches (HF hub cache; no extra cache dir).
- `load_items` keeps its signature and delegates to `resolve_items`. The duplicated `_load_items`
  helpers in `scripts/{run_character,run_audit,refit_edges,validate_method}.py` are consolidated to
  import the package loader (targeted cleanup serving the goal). `build_dataset.load_curated` routes
  through `resolve_items` too.

### B5. Docs
- README: add a "Datasets" section — how resolution works, the four ref forms (local / name /
  `hf://` / `hf-dataset:`), the upload command, and that large sets now live on HF.

---

## Testing

- **Unit (CI-safe, no network):** `resolve_items` parsing for each ref form, using a monkeypatched
  loader/`hf_hub_download` so no real download happens. Local-YAML path covered by existing tiny
  in-repo sets.
- **Integration (manual, networked):** after upload, `load_items("items_2000")` and
  `load_items("config/datasets/items_2000.yaml")` (file absent) both return 2000 items from HF; an
  external probe e.g. `load_items("hf-dataset:Shengtao/recipe:train:title")` returns rows.
- **Rebrand:** full `pytest` green post-rename; `import question_consistency` works.

## Sequencing / deliverables

1. Branch off `main`.
2. Commit 1: rebrand (mechanical + tests green).
3. Commit 2: loader generalization + uploader + tests + README; **then run the uploader** to publish
   to HF.
4. Commit 3: delete the three large YAMLs from git (now on HF), confirm an end-to-end run still
   resolves them from HF.
5. `gh repo rename` + remote URL update.
6. PR.

## Out of scope (YAGNI)

- Renaming the HF logs repo.
- Converting the small in-repo example sets to parquet (they stay YAML).
- Per-item metadata columns on HF (leetcode/recipes difficulty/category) — those sets stay in-repo
  with their `meta` block; only the plain concept pools go to HF.
- Auth/private-dataset handling beyond what `huggingface_hub` already does via the ambient token.
