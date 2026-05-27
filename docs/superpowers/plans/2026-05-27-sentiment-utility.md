# Sentiment Utility Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:codex-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that elicits Gemma-3-12B's sentiment toward 25 "things" via forced-choice logprob comparisons, fits a Thurstonian utility model, and measures coherence/transitivity/completeness — a concepts/objects variant of the Utility Engineering paper.

**Architecture:** Pure-Python package. Model I/O and elicitation in `src/elicit.py`; Thurstonian fit in `src/thurstone.py`; graph metrics in `src/metrics.py`; plotting in `src/plots.py`; orchestration + logging in `src/run.py`. Heavy model code is GPU-only (runs on RunPod); everything else (fit, metrics, parsing, combination) is CPU-testable with synthetic data so the full test suite runs without a GPU.

**Tech Stack:** Python 3.11, UV, PyTorch, transformers, numpy, pandas, seaborn, pyyaml, pytest. Model: `google/gemma-3-12b-it` in bf16.

---

## File Structure

- `pyproject.toml` — UV project, deps, pytest config.
- `config/items.yaml` — 25 items.
- `config/run.yaml` — model id, batch size, seeds, sampling params, fit hyperparams, validation budget.
- `src/sentiment_utility/__init__.py`
- `src/sentiment_utility/data.py` — config loading, item dataclass.
- `src/sentiment_utility/prompts.py` — prompt building + `<answer>` parsing (no torch; CPU-testable).
- `src/sentiment_utility/preferences.py` — ordered→unordered combination, matrix helpers (numpy; CPU-testable).
- `src/sentiment_utility/elicit.py` — model load, batched logprob elicitation, generation validation (torch; GPU).
- `src/sentiment_utility/thurstone.py` — Thurstonian fit + test accuracy (torch; CPU-testable on synthetic).
- `src/sentiment_utility/metrics.py` — transitivity, completeness (numpy; CPU-testable).
- `src/sentiment_utility/plots.py` — seaborn → PDF.
- `src/sentiment_utility/run.py` — orchestrate full run, run folder, config dump, commit hash, save artifacts.
- `tests/` — unit tests for prompts, preferences, thurstone (synthetic recovery), metrics.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/sentiment_utility/__init__.py`, `config/items.yaml`, `config/run.yaml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "sentiment-utility"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "torch",
  "transformers>=4.50",
  "accelerate",
  "numpy",
  "pandas",
  "seaborn",
  "matplotlib",
  "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sentiment_utility"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/sentiment_utility/__init__.py`** (empty file).

- [ ] **Step 3: Create `config/items.yaml`**

```yaml
items:
  - Ronald Reagan
  - Mahatma Gandhi
  - Taylor Swift
  - spaghetti
  - kale
  - chocolate
  - dolphins
  - mosquitoes
  - golden retrievers
  - freedom
  - war
  - taxes
  - democracy
  - laptops
  - plastic bags
  - books
  - the ocean
  - traffic jams
  - sunshine
  - nuclear weapons
  - coffee
  - hospitals
  - spiders
  - music
  - climate change
```

- [ ] **Step 4: Create `config/run.yaml`**

```yaml
model_id: google/gemma-3-12b-it
dtype: bfloat16
batch_size: 64
seed: 0
fit:
  lr: 0.05
  steps: 2000
  test_frac: 0.2
  l2_sigma: 0.01
validation:
  n_pairs: 30
  n_samples: 10
  temperature: 1.0
  max_new_tokens: 16
```

- [ ] **Step 5: Set up env and commit**

Run: `uv sync --extra dev` (expect: venv created, deps resolved). Then:
```bash
git add pyproject.toml src/sentiment_utility/__init__.py config/items.yaml config/run.yaml uv.lock
git commit -m "chore: project scaffold, items and run config"
```

---

### Task 2: Config / data loading

**Files:**
- Create: `src/sentiment_utility/data.py`, `tests/test_data.py`

- [ ] **Step 1: Write failing test** in `tests/test_data.py`

```python
from sentiment_utility.data import load_items, load_run_config

def test_load_items_default():
    items = load_items("config/items.yaml")
    assert len(items) == 25
    assert "spaghetti" in items
    assert len(set(items)) == 25  # unique

def test_load_run_config():
    cfg = load_run_config("config/run.yaml")
    assert cfg["model_id"] == "google/gemma-3-12b-it"
    assert cfg["fit"]["test_frac"] == 0.2
```

- [ ] **Step 2: Run, expect fail**

Run: `uv run pytest tests/test_data.py -v` → FAIL (module not found).

- [ ] **Step 3: Implement `src/sentiment_utility/data.py`**

```python
from __future__ import annotations
import yaml

def load_items(path: str) -> list[str]:
    with open(path) as f:
        data = yaml.safe_load(f)
    items = data["items"]
    if len(items) != len(set(items)):
        raise ValueError("items must be unique")
    return items

def load_run_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Run, expect pass.** `uv run pytest tests/test_data.py -v`

- [ ] **Step 5: Commit**
```bash
git add src/sentiment_utility/data.py tests/test_data.py
git commit -m "feat: config and item loading"
```

---

### Task 3: Prompt building and answer parsing

**Files:**
- Create: `src/sentiment_utility/prompts.py`, `tests/test_prompts.py`

- [ ] **Step 1: Write failing test** in `tests/test_prompts.py`

```python
from sentiment_utility.prompts import build_prompt, parse_answer

def test_build_prompt_contains_items_and_format():
    p = build_prompt("Ronald Reagan", "spaghetti")
    assert "A: Ronald Reagan" in p
    assert "B: spaghetti" in p
    assert "<answer>A</answer>" in p

def test_parse_answer_basic():
    assert parse_answer("<answer>A</answer>") == "A"
    assert parse_answer("blah <answer>B</answer> blah") == "B"

def test_parse_answer_lenient_and_invalid():
    assert parse_answer("I think A") == "A"          # fallback: lone letter
    assert parse_answer("answer: b") == "B"
    assert parse_answer("I cannot choose") is None    # refusal/malformed
    assert parse_answer("<answer>A</answer> <answer>B</answer>") is None  # ambiguous
```

- [ ] **Step 2: Run, expect fail.** `uv run pytest tests/test_prompts.py -v`

- [ ] **Step 3: Implement `src/sentiment_utility/prompts.py`**

```python
from __future__ import annotations
import re

PROMPT_TEMPLATE = (
    "Do you feel more positively about A: {a} or B: {b}? "
    "Answer with <answer>A</answer> or <answer>B</answer>."
)

ASSISTANT_PREFIX = "<answer>"

def build_prompt(a: str, b: str) -> str:
    return PROMPT_TEMPLATE.format(a=a, b=b)

_TAG = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)

def parse_answer(text: str) -> str | None:
    tags = _TAG.findall(text)
    if len(tags) == 1:
        return tags[0].upper()
    if len(tags) > 1:
        return None
    # Fallback: a single unambiguous A or B letter token
    letters = re.findall(r"\b([AB])\b", text, re.IGNORECASE)
    letters = [c.upper() for c in letters]
    uniq = set(letters)
    if len(uniq) == 1:
        return uniq.pop()
    return None
```

Note: when prefilling with `ASSISTANT_PREFIX = "<answer>"`, generated text is concatenated as
`"<answer>" + generation`, so `parse_answer` is called on the full reconstructed string.

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit**
```bash
git add src/sentiment_utility/prompts.py tests/test_prompts.py
git commit -m "feat: prompt building and answer parsing"
```

---

### Task 4: Preference combination and matrices

**Files:**
- Create: `src/sentiment_utility/preferences.py`, `tests/test_preferences.py`

Definitions:
- `ordered_prob[(i, j)]` = P(model picks item i) when prompt shows `A=item i, B=item j` (i≠j).
- `combined_pref[i, j]` = P(i ≻ j), position-bias corrected:
  `0.5 * (ordered_prob[(i,j)] + (1 - ordered_prob[(j,i)]))`.

- [ ] **Step 1: Write failing test** in `tests/test_preferences.py`

```python
import numpy as np
from sentiment_utility.preferences import combine_orderings

def test_combine_orderings_symmetry_and_values():
    # 2 items. When A=0,B=1 model picks item0 with prob 0.8.
    # When A=1,B=0 model picks item1 with prob 0.6 -> picks item0 with 0.4.
    ordered = {(0, 1): 0.8, (1, 0): 0.6}
    pref = combine_orderings(2, ordered)
    # P(0>1) = 0.5*(0.8 + (1-0.6)) = 0.6
    assert np.isclose(pref[0, 1], 0.6)
    # anti-symmetry
    assert np.isclose(pref[1, 0], 0.4)
    assert np.isclose(pref[0, 0], 0.5)  # diagonal convention
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `src/sentiment_utility/preferences.py`**

```python
from __future__ import annotations
import numpy as np

def combine_orderings(n: int, ordered: dict[tuple[int, int], float]) -> np.ndarray:
    pref = np.full((n, n), 0.5, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_ij = ordered[(i, j)]   # P(pick i | A=i, B=j)
            p_ji = ordered[(j, i)]   # P(pick j | A=j, B=i)
            pref[i, j] = 0.5 * (p_ij + (1.0 - p_ji))
    return pref
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit**
```bash
git add src/sentiment_utility/preferences.py tests/test_preferences.py
git commit -m "feat: position-bias-corrected preference combination"
```

---

### Task 5: Thurstonian utility model

**Files:**
- Create: `src/sentiment_utility/thurstone.py`, `tests/test_thurstone.py`

- [ ] **Step 1: Write failing test (synthetic recovery)** in `tests/test_thurstone.py`

```python
import numpy as np
import torch
from scipy.stats import norm  # if scipy unavailable, use math.erf-based cdf in test
from sentiment_utility.thurstone import fit_thurstone, predict_pref_matrix

def _make_synthetic(n=8, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=n)
    sigma = np.full(n, 0.5)
    P = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i, j] = norm.cdf((mu[i]-mu[j]) / np.sqrt(sigma[i]**2 + sigma[j]**2))
    return mu, sigma, P

def test_recovers_ranking():
    mu, sigma, P = _make_synthetic()
    result = fit_thurstone(P, lr=0.1, steps=3000, seed=0)
    fitted = result["mu"]
    # Spearman-style: ranking of fitted mu matches ranking of true mu
    assert np.array_equal(np.argsort(fitted), np.argsort(mu))

def test_predict_matrix_matches_data():
    mu, sigma, P = _make_synthetic()
    result = fit_thurstone(P, lr=0.1, steps=3000, seed=0)
    Phat = predict_pref_matrix(result["mu"], result["sigma"])
    off = ~np.eye(len(mu), dtype=bool)
    assert np.mean(np.abs(Phat[off] - P[off])) < 0.05
```

(If scipy is not a dependency, implement the normal CDF in the test via `math.erf`.)

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `src/sentiment_utility/thurstone.py`**

```python
from __future__ import annotations
import numpy as np
import torch

_NORMAL = torch.distributions.Normal(0.0, 1.0)

def _phi(x: torch.Tensor) -> torch.Tensor:
    return _NORMAL.cdf(x)

def predict_pref_matrix(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    sig_t = torch.as_tensor(sigma, dtype=torch.float64)
    diff = mu_t[:, None] - mu_t[None, :]
    denom = torch.sqrt(sig_t[:, None] ** 2 + sig_t[None, :] ** 2)
    P = _phi(diff / denom)
    P.fill_diagonal_(0.5)
    return P.numpy()

def fit_thurstone(pref: np.ndarray, lr: float = 0.05, steps: int = 2000,
                  test_frac: float = 0.0, l2_sigma: float = 0.01,
                  seed: int = 0) -> dict:
    torch.manual_seed(seed)
    n = pref.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.as_tensor(pref, dtype=torch.float64, device=device)

    # off-diagonal mask, optional train/test split over ordered pairs
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    idx = mask.nonzero(as_tuple=False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(idx.shape[0], generator=g)
    n_test = int(test_frac * idx.shape[0])
    test_idx = idx[perm[:n_test]]
    train_idx = idx[perm[n_test:]]

    train_mask = torch.zeros_like(mask)
    train_mask[train_idx[:, 0], train_idx[:, 1]] = True

    mu = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    log_sigma = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        sigma = torch.exp(log_sigma)
        diff = mu[:, None] - mu[None, :]
        denom = torch.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
        p = _phi(diff / denom).clamp(1e-6, 1 - 1e-6)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        loss = bce[train_mask].mean() + l2_sigma * (log_sigma ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sigma = torch.exp(log_sigma)
        mu_c = mu - mu.mean()  # center for identifiability
        Phat = predict_pref_matrix(mu_c.cpu().numpy(), sigma.cpu().numpy())
        # test accuracy: thresholded predicted vs empirical on held-out (or all) pairs
        eval_idx = test_idx if n_test > 0 else idx
        ph = torch.as_tensor(Phat, device=device)
        pred_label = (ph[eval_idx[:, 0], eval_idx[:, 1]] > 0.5).double()
        emp_label = (target[eval_idx[:, 0], eval_idx[:, 1]] > 0.5).double()
        acc = (pred_label == emp_label).double().mean().item()

    return {
        "mu": mu_c.detach().cpu().numpy(),
        "sigma": sigma.detach().cpu().numpy(),
        "test_accuracy": acc,
        "pred_matrix": Phat,
    }
```

Add `scipy` to dev deps if the test uses it (preferred: implement cdf via `math.erf` in test to
avoid the dep).

- [ ] **Step 4: Run, expect pass.** `uv run pytest tests/test_thurstone.py -v`

- [ ] **Step 5: Commit**
```bash
git add src/sentiment_utility/thurstone.py tests/test_thurstone.py
git commit -m "feat: Thurstonian utility fit with test accuracy"
```

---

### Task 6: Transitivity and completeness metrics

**Files:**
- Create: `src/sentiment_utility/metrics.py`, `tests/test_metrics.py`

- [ ] **Step 1: Write failing test** in `tests/test_metrics.py`

```python
import numpy as np
from sentiment_utility.metrics import (
    cyclic_triad_fraction, expected_cycle_probability, completeness,
)

def test_transitive_chain_has_no_cycles():
    # Perfectly transitive: item i strictly preferred over j iff i>j
    n = 5
    P = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i > j: P[i, j] = 1.0
            elif i < j: P[i, j] = 0.0
    assert cyclic_triad_fraction(P) == 0.0
    assert expected_cycle_probability(P) < 1e-9
    assert np.isclose(completeness(P), 1.0)  # all decisive

def test_indifference_is_incomplete():
    P = np.full((4, 4), 0.5)
    assert np.isclose(completeness(P), 0.0)

def test_single_cycle_detected():
    # 3-cycle: 0>1, 1>2, 2>0
    P = np.array([
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.0, 0.5],
    ])
    assert cyclic_triad_fraction(P) == 1.0
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `src/sentiment_utility/metrics.py`**

```python
from __future__ import annotations
import itertools
import numpy as np

def completeness(pref: np.ndarray) -> float:
    n = pref.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(np.abs(2 * pref[iu] - 1)))

def _prefers(pref, i, j):
    return pref[i, j] > 0.5

def cyclic_triad_fraction(pref: np.ndarray) -> float:
    n = pref.shape[0]
    total = 0
    cyclic = 0
    for i, j, k in itertools.combinations(range(n), 3):
        total += 1
        # count wins within triad; a cycle = each node beats exactly one other
        wins = {i: 0, j: 0, k: 0}
        for a, b in [(i, j), (j, k), (i, k)]:
            if _prefers(pref, a, b): wins[a] += 1
            else: wins[b] += 1
        if set(wins.values()) == {1}:  # all have exactly one win -> 3-cycle
            cyclic += 1
    return cyclic / total if total else 0.0

def expected_cycle_probability(pref: np.ndarray) -> float:
    n = pref.shape[0]
    probs = []
    for i, j, k in itertools.combinations(range(n), 3):
        p_fwd = pref[i, j] * pref[j, k] * pref[k, i]
        p_bwd = pref[j, i] * pref[k, j] * pref[i, k]
        probs.append(p_fwd + p_bwd)
    return float(np.mean(probs)) if probs else 0.0
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit**
```bash
git add src/sentiment_utility/metrics.py tests/test_metrics.py
git commit -m "feat: transitivity and completeness metrics"
```

---

### Task 7: Model elicitation (GPU)

**Files:**
- Create: `src/sentiment_utility/elicit.py`

No unit test (requires GPU + gated model). It will be exercised on the pod via `run.py`. Keep the
module import-safe without a GPU (import torch/transformers lazily inside functions where heavy).

- [ ] **Step 1: Implement `src/sentiment_utility/elicit.py`**

```python
from __future__ import annotations
import itertools
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompts import build_prompt, parse_answer, ASSISTANT_PREFIX

def load_model(model_id: str, dtype: str = "bfloat16"):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="cuda"
    )
    model.eval()
    return tok, model

def _prefill_ids(tok, a: str, b: str) -> str:
    messages = [{"role": "user", "content": build_prompt(a, b)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + ASSISTANT_PREFIX

def _ab_token_ids(tok) -> tuple[int, int]:
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]
    return a_id, b_id

@torch.no_grad()
def elicit_logprobs(tok, model, items: list[str], batch_size: int = 64) -> dict:
    """Return ordered dict {(i,j): P(pick item i | A=i, B=j)} over all i!=j."""
    n = len(items)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    a_id, b_id = _ab_token_ids(tok)
    ordered: dict[tuple[int, int], float] = {}
    for s in range(0, len(pairs), batch_size):
        batch = pairs[s:s + batch_size]
        texts = [_prefill_ids(tok, items[i], items[j]) for (i, j) in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = model(**enc).logits[:, -1, :]          # last position (left-padded)
        ab = torch.stack([logits[:, a_id], logits[:, b_id]], dim=-1)
        p_a = torch.softmax(ab.float(), dim=-1)[:, 0].cpu().numpy()
        for (i, j), pa in zip(batch, p_a):
            ordered[(i, j)] = float(pa)
    return ordered

@torch.no_grad()
def validate_generation(tok, model, items, n_pairs=30, n_samples=10,
                        temperature=1.0, max_new_tokens=16, seed=0) -> dict:
    """Sample real generations for random ordered pairs; return per-pair gen P(pick A) + raw."""
    n = len(items)
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    chosen = [all_pairs[k] for k in rng.choice(len(all_pairs), size=min(n_pairs, len(all_pairs)), replace=False)]
    results = []
    for (i, j) in chosen:
        text = _prefill_ids(tok, items[i], items[j])
        enc = tok([text] * n_samples, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, do_sample=True, temperature=temperature,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        picks = [parse_answer(ASSISTANT_PREFIX + g) for g in gen]
        a_votes = sum(1 for p in picks if p == "A")
        valid = sum(1 for p in picks if p in ("A", "B"))
        results.append({
            "i": i, "j": j, "item_a": items[i], "item_b": items[j],
            "gen_p_a": (a_votes / valid) if valid else None,
            "valid": valid, "n_samples": n_samples, "raw": gen,
        })
    return {"pairs": results}
```

- [ ] **Step 2: Sanity import check (CPU ok)**

Run: `uv run python -c "import sentiment_utility.elicit"`
Expected: no error (heavy load happens only when functions are called).

- [ ] **Step 3: Commit**
```bash
git add src/sentiment_utility/elicit.py
git commit -m "feat: batched logprob elicitation and generation validation"
```

---

### Task 8: Plotting

**Files:**
- Create: `src/sentiment_utility/plots.py`

- [ ] **Step 1: Implement `src/sentiment_utility/plots.py`**

```python
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_sentiment_ranking(items, mu, sigma, path):
    order = np.argsort(mu)
    df = pd.DataFrame({"item": [items[i] for i in order],
                       "mu": mu[order], "sigma": sigma[order]})
    plt.figure(figsize=(8, 9))
    ax = sns.barplot(data=df, y="item", x="mu", color="#4C72B0")
    ax.errorbar(df["mu"], range(len(df)), xerr=df["sigma"], fmt="none",
                ecolor="gray", capsize=2)
    ax.set_xlabel("Thurstonian utility μ (sentiment)")
    ax.set_ylabel("")
    ax.set_title("Gemma-3-12B sentiment ranking")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_validation_scatter(logprob_p_a, gen_p_a, path):
    df = pd.DataFrame({"logprob_p_a": logprob_p_a, "gen_p_a": gen_p_a}).dropna()
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=df, x="logprob_p_a", y="gen_p_a")
    ax.plot([0, 1], [0, 1], ls="--", color="gray")
    r = df.corr().iloc[0, 1] if len(df) > 1 else float("nan")
    ax.set_title(f"Logprob vs generation P(pick A)  (r={r:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_preference_heatmap(items, pref, mu, path):
    order = np.argsort(-mu)
    M = pref[np.ix_(order, order)]
    labels = [items[i] for i in order]
    plt.figure(figsize=(10, 9))
    sns.heatmap(M, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1,
                cmap="RdBu_r", cbar_kws={"label": "P(row ≻ col)"})
    plt.title("Pairwise preference matrix (sorted by μ)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
```

- [ ] **Step 2: Sanity check**

Run: `uv run python -c "import sentiment_utility.plots"` → no error.

- [ ] **Step 3: Commit**
```bash
git add src/sentiment_utility/plots.py
git commit -m "feat: seaborn PDF plots"
```

---

### Task 9: Orchestration, logging, run folder

**Files:**
- Create: `src/sentiment_utility/run.py`

- [ ] **Step 1: Implement `src/sentiment_utility/run.py`**

```python
from __future__ import annotations
import json, subprocess, datetime, logging, sys
from pathlib import Path
import numpy as np

from .data import load_items, load_run_config
from .preferences import combine_orderings
from .elicit import load_model, elicit_logprobs, validate_generation
from .thurstone import fit_thurstone
from .metrics import cyclic_triad_fraction, expected_cycle_probability, completeness
from .plots import plot_sentiment_ranking, plot_validation_scatter, plot_preference_heatmap

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def main(items_path="config/items.yaml", run_path="config/run.yaml"):
    items = load_items(items_path)
    cfg = load_run_config(run_path)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(run_dir / "run.log"), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)
    commit = _git_commit()
    log.info("commit=%s", commit)
    (run_dir / "config.json").write_text(json.dumps(
        {"commit": commit, "items": items, "run_config": cfg}, indent=2))

    log.info("loading model %s", cfg["model_id"])
    tok, model = load_model(cfg["model_id"], cfg["dtype"])

    log.info("eliciting %d ordered pairs via logprobs", len(items) * (len(items) - 1))
    ordered = elicit_logprobs(tok, model, items, batch_size=cfg["batch_size"])
    pref = combine_orderings(len(items), ordered)
    np.save(run_dir / "pref_matrix.npy", pref)
    json.dump({f"{i}_{j}": v for (i, j), v in ordered.items()},
              open(run_dir / "ordered_probs.json", "w"), indent=2)

    log.info("fitting Thurstonian model")
    fit = fit_thurstone(pref, lr=cfg["fit"]["lr"], steps=cfg["fit"]["steps"],
                        test_frac=cfg["fit"]["test_frac"], l2_sigma=cfg["fit"]["l2_sigma"],
                        seed=cfg["seed"])
    mu, sigma = fit["mu"], fit["sigma"]

    metrics = {
        "utility_test_accuracy": fit["test_accuracy"],
        "cyclic_triad_fraction": cyclic_triad_fraction(pref),
        "expected_cycle_probability": expected_cycle_probability(pref),
        "completeness": completeness(pref),
    }
    log.info("metrics=%s", json.dumps(metrics))

    log.info("running generation validation")
    val = validate_generation(tok, model, items,
                              n_pairs=cfg["validation"]["n_pairs"],
                              n_samples=cfg["validation"]["n_samples"],
                              temperature=cfg["validation"]["temperature"],
                              max_new_tokens=cfg["validation"]["max_new_tokens"],
                              seed=cfg["seed"])
    # align logprob P(pick A) for validated pairs
    for r in val["pairs"]:
        r["logprob_p_a"] = ordered[(r["i"], r["j"])]
    lp = [r["logprob_p_a"] for r in val["pairs"]]
    gp = [r["gen_p_a"] for r in val["pairs"]]
    valid_pairs = [(a, b) for a, b in zip(lp, gp) if b is not None]
    if len(valid_pairs) > 1:
        a_arr, b_arr = np.array([p[0] for p in valid_pairs]), np.array([p[1] for p in valid_pairs])
        metrics["validation_pearson_r"] = float(np.corrcoef(a_arr, b_arr)[0, 1])
        metrics["validation_agreement"] = float(np.mean((a_arr > 0.5) == (b_arr > 0.5)))
    metrics["validation_malformed_rate"] = float(np.mean(
        [1 - r["valid"] / r["n_samples"] for r in val["pairs"]]))

    # ranking output
    order = np.argsort(-mu)
    ranking = [{"item": items[i], "mu": float(mu[i]), "sigma": float(sigma[i])} for i in order]

    json.dump({"metrics": metrics, "ranking": ranking, "validation": val["pairs"]},
              open(run_dir / "results.json", "w"), indent=2)

    plot_sentiment_ranking(items, mu, sigma, run_dir / "sentiment_ranking.pdf")
    plot_preference_heatmap(items, pref, mu, run_dir / "preference_heatmap.pdf")
    plot_validation_scatter(lp, gp, run_dir / "validation_scatter.pdf")
    log.info("done -> %s", run_dir)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity import check.** `uv run python -c "import sentiment_utility.run"` → no error.

- [ ] **Step 3: Commit**
```bash
git add src/sentiment_utility/run.py
git commit -m "feat: run orchestration with logging and artifact saving"
```

---

### Task 10: Full local test suite + README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run full suite.** `uv run pytest -v` → all CPU tests pass.

- [ ] **Step 2: Write `README.md`** documenting: purpose, install (`uv sync`), run
  (`uv run python -m sentiment_utility.run`), GPU requirement (~48GB, gated Gemma access via
  `huggingface-cli login`), and outputs (`runs/<ts>/`).

- [ ] **Step 3: Commit**
```bash
git add README.md
git commit -m "docs: README with usage and GPU requirements"
```

---

## Self-Review Notes

- **Spec coverage:** items (T1), logprob elicitation all 600 ordered pairs (T7), position-bias
  combination (T4), Thurstonian fit + test accuracy (T5), transitivity + completeness (T6),
  generation validation 30×10 (T7/T9), plots (T8), logging/run folder/commit hash (T9). ✓
- **GPU vs CPU:** all math/parsing modules are CPU-testable; only `elicit.py` needs GPU and is
  exercised via the pod run. ✓
- **Type consistency:** `ordered` keyed by `(i, j)` tuples everywhere; `fit_thurstone` returns
  `mu/sigma/test_accuracy/pred_matrix`; `combine_orderings(n, ordered)` signature consistent across
  T4/T7/T9. ✓
- **Out of scope:** activation linear-probe excluded (too few items), matching the spec. ✓
```
