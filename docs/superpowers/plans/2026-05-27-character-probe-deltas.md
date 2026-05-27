# Character-Model Sentiment Probes & Deltas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:codex-driven-development. Steps use checkbox syntax.

**Goal:** Apply the sentiment pipeline to Open Character Training Llama-3.1-8B models: per-model probe (trained on 500-concept elicited μ), KV-cached probe-scoring of a 2000-concept set, and base-vs-character sentiment deltas. Base + Loving first, then loop personas.

**Architecture:** Reuse efficient.py/thurstone.py/probe.py. Add: `peft` adapter loading, probe persistence + deployable fit, KV-cached `probe_score_concepts`, delta/z-score analysis, orchestration scripts.

**Tech Stack:** existing + `peft`. Models: `meta-llama/Llama-3.1-8B-Instruct` + `maius/llama-3.1-8b-it-personas` (subfolders) + `maius/llama-3.1-8b-it-misalignment`.

---

## File Structure
- `src/sentiment_utility/characters.py` — model registry + `load_character_model`.
- `src/sentiment_utility/probe.py` — add `fit_deployable_probe`, `save_probe`, `load_probe`, `apply_probe`, `common_token_prefix`, `probe_score_concepts`.
- `src/sentiment_utility/deltas.py` — `zscore`, `score_deltas`.
- `scripts/build_dataset.py` — add argparse (`--n`, `--out`, quotas) to also build `config/items_2000.yaml`.
- `scripts/run_character.py`, `scripts/run_all_characters.py`, `scripts/compare_characters.py`.
- Tests: `tests/test_probe_persistence.py`, `tests/test_deltas.py`, `tests/test_characters.py`.

---

### Task 0: peft dependency + 2000-concept dataset
- [ ] Add `"peft"` to `pyproject.toml` deps; `uv sync --extra dev`.
- [ ] Add argparse to `scripts/build_dataset.py`: `--n` (default 500), `--out` (default config/items_500.yaml), and per-source quota flags (defaults curated=250, things=150, warriner=N-250). Keep existing behaviour when run with no args. Ensure the 2000-set is a deterministic superset-ish (same seed) so the 500 train concepts are included where possible: when building 2000, set warriner quota to 1750.
- [ ] Commit: `chore: add peft; parametrize build_dataset for items_2000`.

---

### Task 1: Character model registry + loader
**Files:** `src/sentiment_utility/characters.py`, `tests/test_characters.py`

Contract:
```python
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PERSONAS_REPO = "maius/llama-3.1-8b-it-personas"
MISALIGNMENT_REPO = "maius/llama-3.1-8b-it-misalignment"
PERSONA_SUBFOLDERS = ["loving","goodness","humor","sarcasm","poeticism","mathematical",
                      "nonchalance","impulsiveness","remorse","sycophancy"]

def model_specs() -> list[dict]:
    """Return [{"name","repo"|None,"subfolder"|None}] for base + all 11 personas + misalignment."""

def load_character_model(spec, dtype="bfloat16"):
    """Load base via elicit.load_model; if spec has a repo, wrap with PeftModel.from_pretrained."""
```

- [ ] **Step 1: Test** `tests/test_characters.py` (pure registry — no GPU):
```python
from sentiment_utility.characters import model_specs, PERSONA_SUBFOLDERS

def test_specs_include_base_and_all_personas():
    specs = model_specs()
    names = [s["name"] for s in specs]
    assert "base" in names
    assert "loving" in names
    assert "misalignment" in names
    assert len([s for s in specs if s["name"] != "base"]) == 11   # 10 personas + misalignment
    base = next(s for s in specs if s["name"] == "base")
    assert base["repo"] is None
    loving = next(s for s in specs if s["name"] == "loving")
    assert loving["repo"].endswith("personas") and loving["subfolder"] == "loving"
```
- [ ] **Step 2:** run, fail.
- [ ] **Step 3: Implement** `characters.py`. `model_specs()` = base (repo None) + one entry per PERSONA_SUBFOLDERS (repo=PERSONAS_REPO, subfolder=name) + misalignment (repo=MISALIGNMENT_REPO, subfolder=None). `load_character_model(spec, dtype)`: `tok, model = load_model(BASE_MODEL, dtype)`; if `spec["repo"]`: `from peft import PeftModel; model = PeftModel.from_pretrained(model, spec["repo"], subfolder=spec.get("subfolder"))`; `model.eval()`; return tok, model. Import peft lazily inside the function.
- [ ] **Step 4:** run, pass.
- [ ] **Step 5:** commit `feat: character model registry and PEFT adapter loader`.

---

### Task 2: Probe persistence + deployable fit
**Files:** `src/sentiment_utility/probe.py` (extend), `tests/test_probe_persistence.py`

Add:
```python
def fit_deployable_probe(X, y, alpha=1.0) -> dict:
    # ridge on ALL rows; returns {"coef": (d,), "intercept": float, "alpha": alpha}
def apply_probe(X, probe) -> np.ndarray:        # X @ coef + intercept
def save_probe(path, probe: dict) -> None       # json (lists) incl. "best_layer"
def load_probe(path) -> dict
```

- [ ] **Step 1: Test** `tests/test_probe_persistence.py`:
```python
import numpy as np, tempfile, os
from sentiment_utility.probe import fit_deployable_probe, apply_probe, save_probe, load_probe

def test_fit_apply_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 12)); w = rng.normal(size=12); y = X @ w + 0.01*rng.normal(size=150)
    p = fit_deployable_probe(X, y, alpha=0.1)
    pred = apply_probe(X, p)
    assert np.corrcoef(pred, y)[0,1] > 0.99

def test_save_load_roundtrip(tmp_path):
    p = {"coef": [1.0, 2.0, 3.0], "intercept": 0.5, "alpha": 1.0, "best_layer": 7}
    path = tmp_path / "probe.json"; save_probe(path, p); q = load_probe(path)
    assert q["best_layer"] == 7 and np.allclose(q["coef"], p["coef"])
```
- [ ] **Step 2-4:** fail → implement (Ridge from sklearn; coef/intercept to lists in save; np arrays on load) → pass.
- [ ] **Step 5:** commit `feat: deployable probe fit + save/load`.

---

### Task 3: KV-cache prefix helper + scoring (CPU-testable parts)
**Files:** `src/sentiment_utility/probe.py` (extend), `tests/test_probe_persistence.py` (extend)

Add `common_token_prefix(seqs) -> list[int]` (longest shared prefix of token-id lists) and the GPU
`probe_score_concepts(tok, model, items, best_layer, probe, batch_size=16)` (lazy torch import). The
scoring function: render each concept's chat-template prompt (add_generation_prompt=False) → token
lists; PREFIX = common_token_prefix(a few rendered prompts); run model once on PREFIX → past_kv;
per batch, expand past_kv to batch size and forward the per-row suffix tokens (right-padded) with a
full attention mask, `output_hidden_states=True`, gather best_layer hidden at each row's true last
index; `apply_probe`. Return `np.ndarray (N,)`.

- [ ] **Step 1: Test** the pure helper:
```python
from sentiment_utility.probe import common_token_prefix
def test_common_token_prefix():
    assert common_token_prefix([[1,2,3,9],[1,2,3,8],[1,2,3,7,7]]) == [1,2,3]
    assert common_token_prefix([[5,1],[6,1]]) == []
    assert common_token_prefix([[1,2]]) == [1,2]
```
- [ ] **Step 2-4:** fail → implement both functions → pass (only the helper is unit-tested on CPU; `probe_score_concepts` is GPU-validated via an equivalence check in the run).
- [ ] **Step 5:** commit `feat: KV-cached probe scoring + common-prefix helper`.

---

### Task 4: Delta / z-score analysis
**Files:** `src/sentiment_utility/deltas.py`, `tests/test_deltas.py`

```python
def zscore(x) -> np.ndarray
def score_deltas(items, base_scores, char_scores, top_k=20) -> dict
    # z-scores each, delta = z_char - z_base; returns {"pearson_r", "mean_abs_delta",
    #   "more_positive":[{item,delta}], "more_negative":[...], "delta": {item: float}}
```
- [ ] **Step 1: Test** `tests/test_deltas.py`:
```python
import numpy as np
from sentiment_utility.deltas import zscore, score_deltas

def test_zscore_props():
    z = zscore([1,2,3,4,5]); assert abs(z.mean()) < 1e-9 and abs(z.std()-1) < 1e-9

def test_score_deltas_identifies_shift():
    items = list("abcde")
    base = np.array([0,1,2,3,4.0]); char = base.copy(); char[0] += 10  # 'a' way up for char
    d = score_deltas(items, base, char, top_k=2)
    assert d["more_positive"][0]["item"] == "a"
    assert "pearson_r" in d and "mean_abs_delta" in d
```
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** commit `feat: z-score sentiment delta analysis`.

---

### Task 5: Per-model orchestration
**Files:** `scripts/run_character.py`

- [ ] Implement: args `--spec-name` (base/loving/...); resolve spec via `model_specs()`; load model;
  efficient elicit μ on items_500 (compare_pairs oracle → rank_by_quicksort → spacing_pass →
  fit_thurstone_sparse); extract_activations(500) → probe_all_layers → best layer + R²; fit
  deployable probe on all 500 at best layer, save_probe; **equivalence check**: on a 32-concept
  sample compare `probe_score_concepts` vs `apply_probe(extract_activations[best_layer])` and log max
  abs diff (assert < 1e-2); probe-score items_2000 via `probe_score_concepts`; save
  `runs/character/<name>/` with config+commit, elicited μ, probe.json, metrics, scores_2000.json,
  R²-vs-layer plot. Print summary.
- [ ] Sanity: `uv run python -c "import sentiment_utility.characters, sentiment_utility.deltas"` and
  `ast.parse` the script.
- [ ] Commit `feat: per-character run (elicit, probe, KV-scored 2000)`.

---

### Task 6: Loop + compare
**Files:** `scripts/run_all_characters.py`, `scripts/compare_characters.py`

- [ ] `run_all_characters.py`: iterate `model_specs()` (base first), calling the run_character flow
  (import its `main` or a `run_one(spec)` function — refactor run_character to expose `run_one`).
  Skip models whose run dir already exists (resumable).
- [ ] `compare_characters.py`: load base scores_2000 + each character's scores_2000; `score_deltas`
  per character vs base; write `runs/character/deltas/<name>.json` and plots (base-vs-char scatter,
  top/bottom-Δ bars); a summary table CSV of mean_abs_delta + pearson_r per character.
- [ ] Sanity ast.parse. Commit `feat: loop all characters + delta comparison`.

---

### Task 7: Full suite + README
- [ ] `uv run pytest -v` (all pass on CPU).
- [ ] README section on the character-delta workflow.
- [ ] Commit `docs: character probe/delta workflow`.

---

## Self-Review Notes
- Spec coverage: peft+2000 set (T0), adapter load (T1), probe persistence (T2), KV-scoring (T3),
  deltas (T4), per-model run + equivalence gate (T5), loop+compare (T6). ✓
- Type consistency: `model_specs()` dicts {name,repo,subfolder}; probe dict {coef,intercept,alpha,
  best_layer}; `score_deltas` keys used by compare_characters. ✓
- GPU vs CPU: registry, probe persistence/apply, common_token_prefix, deltas all CPU-tested;
  adapter load + probe_score_concepts equivalence GPU-validated in the run. ✓
- Faithfulness: KV-scored activations must match plain extraction (equivalence gate < 1e-2). ✓
