# Scaling + Efficient Elicitation + Linear Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:codex-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a 500-concept dataset, an O(n log n) transitivity-exploiting elicitation method (validated lossless vs O(n²)), and a linear probe predicting Thurstonian μ from Gemma-3-12B activations.

**Architecture:** New modules `dataset.py` (offline sampler), `efficient.py` (batched-pivot quicksort + sparse Thurstonian fit), `probe.py` (ridge probe + metrics). Network fetching and GPU work live in `scripts/`. All algorithmic logic is CPU-testable with synthetic oracles/activations; only real model comparisons and activation extraction need GPU.

**Tech Stack:** existing stack + scikit-learn (ridge regression). Reuse `elicit.py`, `thurstone.py`, `metrics.py`, `plots.py`.

---

## File Structure
- `src/sentiment_utility/dataset.py` — pure sampler `build_pool_sample`.
- `src/sentiment_utility/efficient.py` — `rank_by_quicksort`, `spacing_pass`, `fit_thurstone_sparse`, `edges_to_implied_matrix`.
- `src/sentiment_utility/probe.py` — `train_probe`, `probe_all_layers`.
- `scripts/build_dataset.py` — fetch THINGS/Warriner, call sampler, write `config/items_500.yaml`.
- `scripts/validate_method.py` — dense vs efficient on a 60-subset.
- `scripts/run_scale.py` — full 500 run (GPU): efficient elicit → sparse fit → metrics → probe → plots.
- `config/curated_concepts.yaml` — already created (250 concepts).
- Tests: `tests/test_dataset.py`, `tests/test_efficient.py`, `tests/test_probe.py`.

Add `scikit-learn` to `pyproject.toml` dependencies (Task 0).

---

### Task 0: Add scikit-learn dependency

- [ ] **Step 1:** Add `"scikit-learn"` to `[project].dependencies` in `pyproject.toml`.
- [ ] **Step 2:** Run `uv sync --extra dev` (expect resolve OK).
- [ ] **Step 3:** Commit: `git add pyproject.toml uv.lock && git commit -m "chore: add scikit-learn for linear probe"`

---

### Task 1: Dataset sampler (pure, offline)

**Files:** Create `src/sentiment_utility/dataset.py`, `tests/test_dataset.py`

Contract: `build_pool_sample(sources, quotas, n, seed) -> (items, meta)` where
- `sources: dict[str, list[tuple[str, float|None]]]` maps source name → list of `(name, human_valence)`.
- `quotas: dict[str, int]` desired count per source (may exceed availability).
- Dedupe case-insensitively across ALL sources (first occurrence by source order wins).
- Deterministically sample up to each quota per source; if total < n, top up from remaining pooled
  items (any source) deterministically; if total > n, trim deterministically. Return exactly
  `min(n, total_unique)` items.
- `items: list[str]`; `meta: dict[str, dict]` name → `{"source": str, "human_valence": float|None}`.

- [ ] **Step 1: Write failing test** `tests/test_dataset.py`

```python
from sentiment_utility.dataset import build_pool_sample

def _sources():
    return {
        "curated": [(f"c{i}", None) for i in range(10)],
        "things":  [(f"t{i}", None) for i in range(10)],
        "warriner":[(f"w{i}", float(i)) for i in range(10)],
    }

def test_exact_n_and_dedupe():
    items, meta = build_pool_sample(_sources(), {"curated":5,"things":5,"warriner":5}, n=12, seed=0)
    assert len(items) == 12
    assert len(set(items)) == 12
    for it in items:
        assert meta[it]["source"] in {"curated","things","warriner"}

def test_dedupe_across_sources_first_wins():
    src = {"curated": [("apple", None)], "warriner": [("Apple", 5.0)]}
    items, meta = build_pool_sample(src, {"curated":1,"warriner":1}, n=2, seed=0)
    assert len(items) == 1                 # case-insensitive dedupe
    assert meta[items[0]]["source"] == "curated"   # first source wins

def test_determinism():
    a = build_pool_sample(_sources(), {"curated":5,"things":5,"warriner":5}, n=12, seed=1)[0]
    b = build_pool_sample(_sources(), {"curated":5,"things":5,"warriner":5}, n=12, seed=1)[0]
    assert a == b

def test_topup_when_quota_exceeds_pool():
    items, _ = build_pool_sample(_sources(), {"curated":100,"things":0,"warriner":0}, n=25, seed=0)
    assert len(items) == 25   # 10 curated + top-up 15 from others
```

- [ ] **Step 2: Run, expect fail.** `uv run pytest tests/test_dataset.py -v`

- [ ] **Step 3: Implement `src/sentiment_utility/dataset.py`**

```python
from __future__ import annotations
import random

def build_pool_sample(sources, quotas, n, seed=0):
    rng = random.Random(seed)
    seen = set()                       # lowercased names already taken
    pooled = []                        # (name, source, valence) deduped, source-order
    chosen, meta = [], {}

    def add_unique(name, source, valence, bucket):
        key = name.strip().lower()
        if not name.strip() or key in seen:
            return
        seen.add(key)
        bucket.append((name, source, valence))

    # dedupe within each source, preserving source order for "first wins"
    per_source = {}
    for source, entries in sources.items():
        bucket = []
        for name, valence in entries:
            add_unique(name, source, valence, bucket)
        per_source[source] = bucket
        pooled.extend(bucket)

    # sample up to quota per source
    for source, bucket in per_source.items():
        q = quotas.get(source, 0)
        picks = bucket if q >= len(bucket) else rng.sample(bucket, q)
        for name, src, val in picks:
            if name not in meta:
                chosen.append(name)
                meta[name] = {"source": src, "human_valence": val}

    # top up from remaining pooled items if short of n
    if len(chosen) < n:
        remaining = [t for t in pooled if t[0] not in meta]
        rng.shuffle(remaining)
        for name, src, val in remaining:
            if len(chosen) >= n:
                break
            chosen.append(name)
            meta[name] = {"source": src, "human_valence": val}

    # trim deterministically if over n
    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
        meta = {k: meta[k] for k in chosen}

    return chosen, meta
```

- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit:** `git add src/sentiment_utility/dataset.py tests/test_dataset.py && git commit -m "feat: deterministic multi-source concept pool sampler"`

---

### Task 2: Efficient sorter + sparse Thurstonian fit

**Files:** Create `src/sentiment_utility/efficient.py`, `tests/test_efficient.py`

Contracts:
- `rank_by_quicksort(n, oracle, seed=0) -> (order, edges)`. `oracle(pairs)` takes a list of
  `(i, j)` and returns `{(i, j): P(prefer i over j)}`. `order` is item indices best→worst (highest
  utility first). `edges` is `list[(i, j, p)]` of every comparison made (p = P(i≻j)).
- `spacing_pass(order, oracle, k=2) -> edges`: compares each item to its next `k` neighbours in
  `order`.
- `edges_to_implied_matrix(mu, sigma) -> np.ndarray`: thin wrapper over `predict_pref_matrix`.
- `fit_thurstone_sparse(edges, n, lr=0.05, steps=2000, test_frac=0.2, l2_sigma=0.01, seed=0) -> dict`
  with keys `mu, sigma, test_accuracy, accuracy_is_heldout, pred_matrix, comparison_count`.

- [ ] **Step 1: Write failing test** `tests/test_efficient.py`

```python
import numpy as np
from math import erf, sqrt
from sentiment_utility.efficient import rank_by_quicksort, spacing_pass, fit_thurstone_sparse

def _phi(x): return 0.5 * (1 + erf(x / sqrt(2)))

def _oracle_factory(mu, sigma=0.4):
    def oracle(pairs):
        out = {}
        for i, j in pairs:
            out[(i, j)] = _phi((mu[i] - mu[j]) / sqrt(2) / sigma)
        return out
    return oracle

def test_quicksort_recovers_order():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=40)
    oracle = _oracle_factory(mu, sigma=0.2)   # low noise -> clean order
    order, edges = rank_by_quicksort(len(mu), oracle, seed=0)
    # order is best->worst; should match descending mu closely (allow few swaps)
    true_desc = list(np.argsort(-mu))
    # Spearman-ish: rank positions correlate strongly
    pos = {idx: r for r, idx in enumerate(order)}
    tpos = {idx: r for r, idx in enumerate(true_desc)}
    diffs = sum(abs(pos[i] - tpos[i]) for i in range(len(mu)))
    assert diffs < len(mu)   # average displacement < 1 position

def test_quicksort_subquadratic_comparison_count():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=64)
    counts = {"n": 0}
    base = _oracle_factory(mu, 0.2)
    def counting(pairs):
        counts["n"] += len(pairs)
        return base(pairs)
    rank_by_quicksort(len(mu), counting, seed=0)
    assert counts["n"] < 64 * 63          # strictly fewer than dense n(n-1)
    assert counts["n"] < 8 * 64           # ~ < c*n*log2(n); generous bound

def test_sparse_fit_recovers_ranking():
    rng = np.random.default_rng(2)
    mu = rng.normal(size=30)
    oracle = _oracle_factory(mu, 0.3)
    order, edges = rank_by_quicksort(len(mu), oracle, seed=0)
    edges += spacing_pass(order, oracle, k=2)
    res = fit_thurstone_sparse(edges, len(mu), steps=3000, seed=0)
    # fitted mu rank-correlates strongly with true mu
    from scipy.stats import spearmanr
    rho = spearmanr(res["mu"], mu).statistic
    assert rho > 0.9
    assert res["comparison_count"] == len(edges)
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `src/sentiment_utility/efficient.py`**

```python
from __future__ import annotations
import numpy as np
import torch
from .thurstone import predict_pref_matrix, _phi  # _phi: torch normal cdf

def rank_by_quicksort(n, oracle, seed=0):
    """Randomized batched-pivot quicksort. Returns (order best->worst, edges[(i,j,p)])."""
    rng = np.random.default_rng(seed)
    edges = []

    def sort(bucket):
        if len(bucket) <= 1:
            return list(bucket)
        pivot = bucket[rng.integers(len(bucket))]
        rest = [x for x in bucket if x != pivot]
        pairs = [(x, pivot) for x in rest]          # P(prefer x over pivot)
        probs = oracle(pairs)
        greater, lesser = [], []                    # greater = higher utility than pivot
        for x in rest:
            p = probs[(x, pivot)]
            edges.append((x, pivot, float(p)))
            (greater if p > 0.5 else lesser).append(x)
        return sort(greater) + [pivot] + sort(lesser)

    order = sort(list(range(n)))
    return order, edges

def spacing_pass(order, oracle, k=2):
    pairs = []
    for r in range(len(order)):
        for d in range(1, k + 1):
            if r + d < len(order):
                pairs.append((order[r], order[r + d]))
    probs = oracle(pairs)
    return [(i, j, float(probs[(i, j)])) for (i, j) in pairs]

def edges_to_implied_matrix(mu, sigma):
    return predict_pref_matrix(mu, sigma)

def fit_thurstone_sparse(edges, n, lr=0.05, steps=2000, test_frac=0.2,
                         l2_sigma=0.01, seed=0):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = len(edges)
    perm = torch.randperm(m, generator=g).tolist()
    n_test = int(test_frac * m)
    test_set = set(perm[:n_test])

    ii = torch.tensor([e[0] for e in edges], device=device)
    jj = torch.tensor([e[1] for e in edges], device=device)
    pp = torch.tensor([e[2] for e in edges], dtype=torch.float64, device=device)
    is_train = torch.tensor([k not in test_set for k in range(m)], device=device)

    mu = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    log_sigma = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        sigma = torch.exp(log_sigma)
        denom = torch.sqrt(sigma[ii] ** 2 + sigma[jj] ** 2)
        p = _phi((mu[ii] - mu[jj]) / denom).clamp(1e-6, 1 - 1e-6)
        bce = -(pp * torch.log(p) + (1 - pp) * torch.log(1 - p))
        train_loss = bce[is_train].mean() if is_train.any() else bce.mean()
        loss = train_loss + l2_sigma * (log_sigma ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sigma = torch.exp(log_sigma)
        scale = sigma.mean()
        mu_c = (mu - mu.mean()) / scale
        sigma_c = sigma / scale
        Phat = predict_pref_matrix(mu_c.cpu().numpy(), sigma_c.cpu().numpy())
        eval_mask = ~is_train if (n_test > 0 and (~is_train).any()) else torch.ones_like(is_train)
        pred = (_phi((mu_c[ii] - mu_c[jj]) / torch.sqrt(sigma_c[ii] ** 2 + sigma_c[jj] ** 2)) > 0.5)
        emp = pp > 0.5
        acc = (pred[eval_mask] == emp[eval_mask]).double().mean().item()

    return {
        "mu": mu_c.detach().cpu().numpy(),
        "sigma": sigma_c.detach().cpu().numpy(),
        "test_accuracy": acc,
        "accuracy_is_heldout": n_test > 0,
        "pred_matrix": Phat,
        "comparison_count": m,
    }
```

Note: `thurstone.py` currently defines `_phi` at module scope — confirm it is importable; if it is
named differently, add `from .thurstone import _phi` equivalent or re-expose it.

- [ ] **Step 4: Run, expect pass.** `uv run pytest tests/test_efficient.py -v`
- [ ] **Step 5: Commit:** `git add src/sentiment_utility/efficient.py tests/test_efficient.py && git commit -m "feat: O(n log n) batched-pivot quicksort + sparse Thurstonian fit"`

---

### Task 3: Linear probe

**Files:** Create `src/sentiment_utility/probe.py`, `tests/test_probe.py`

Contracts:
- `train_probe(X, y, seed=0, alpha=1.0, test_frac=0.2) -> dict` with keys `test_r2`,
  `pairwise_accuracy` (sign agreement of y_i−y_j over held-out pairs), `n_test`.
- `probe_all_layers(hidden, y, seed=0, alpha=1.0) -> dict` where `hidden: dict[int, np.ndarray]`
  layer → (N, d); returns `{"per_layer": {layer: train_probe(...)}, "best_layer": int,
  "best_r2": float}`.

- [ ] **Step 1: Write failing test** `tests/test_probe.py`

```python
import numpy as np
from sentiment_utility.probe import train_probe, probe_all_layers

def test_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    w = rng.normal(size=16)
    y = X @ w + 0.01 * rng.normal(size=200)    # near-perfect linear signal
    res = train_probe(X, y, seed=0, alpha=0.1)
    assert res["test_r2"] > 0.95
    assert res["pairwise_accuracy"] > 0.9

def test_noise_layer_scores_low_best_layer_picks_signal():
    rng = np.random.default_rng(1)
    N = 200
    y = rng.normal(size=N)
    signal = np.outer(y, rng.normal(size=8)) + 0.01 * rng.normal(size=(N, 8))
    noise = rng.normal(size=(N, 8))
    res = probe_all_layers({0: noise, 1: signal}, y, seed=0, alpha=0.1)
    assert res["best_layer"] == 1
    assert res["per_layer"][1]["test_r2"] > res["per_layer"][0]["test_r2"]
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `src/sentiment_utility/probe.py`**

```python
from __future__ import annotations
import numpy as np
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def train_probe(X, y, seed=0, alpha=1.0, test_frac=0.2):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac, random_state=seed)
    model = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred = model.predict(Xte)
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # pairwise sign-agreement on held-out
    agree = total = 0
    for a, b in combinations(range(len(yte)), 2):
        if yte[a] == yte[b]:
            continue
        total += 1
        agree += (pred[a] > pred[b]) == (yte[a] > yte[b])
    return {"test_r2": float(r2),
            "pairwise_accuracy": float(agree / total) if total else float("nan"),
            "n_test": int(len(yte))}

def probe_all_layers(hidden, y, seed=0, alpha=1.0):
    per_layer = {layer: train_probe(X, y, seed=seed, alpha=alpha)
                 for layer, X in hidden.items()}
    best_layer = max(per_layer, key=lambda L: per_layer[L]["test_r2"])
    return {"per_layer": per_layer,
            "best_layer": int(best_layer),
            "best_r2": float(per_layer[best_layer]["test_r2"])}
```

- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit:** `git add src/sentiment_utility/probe.py tests/test_probe.py && git commit -m "feat: ridge linear sentiment probe with per-layer selection"`

---

### Task 4: Dataset build script (network)

**Files:** Create `scripts/build_dataset.py`

- [ ] **Step 1: Implement** a script that:
  - loads `config/curated_concepts.yaml` → curated source (valence None).
  - tries to fetch Warriner CSV (`http://crr.ugent.be/papers/Ratings_Warriner_et_al.csv`); parse
    columns `Word`, `V.Mean.Sum`; on failure, warn and use empty list.
  - tries to fetch a THINGS concept list (try a couple of known raw URLs); take the concept column;
    on failure, warn and use empty list.
  - calls `build_pool_sample(sources, quotas={"curated":250,"things":150,"warriner":100}, n=500,
    seed=0)`.
  - writes `config/items_500.yaml` with `items:` and `meta:` (source + human_valence).
  - prints per-source counts.
  Use `urllib.request` (stdlib) with a timeout; wrap each fetch in try/except.

- [ ] **Step 2: Sanity:** `uv run python -c "import ast"` (script will actually run on the pod with
  network). Locally just confirm it imports: `uv run python -c "import importlib.util as u; u.spec_from_file_location('b','scripts/build_dataset.py')"`.
- [ ] **Step 3: Commit:** `git add scripts/build_dataset.py && git commit -m "feat: dataset build script (curated+THINGS+Warriner -> items_500)"`

---

### Task 5: Method-validation script (GPU)

**Files:** Create `scripts/validate_method.py`

- [ ] **Step 1: Implement** a script that, given a 60-item subset of `items_500.yaml` (first 60 or a
  seeded sample): loads the model once (`elicit.load_model`); runs dense `elicit_logprobs` +
  `combine_orderings` + `fit_thurstone`; runs the efficient pipeline using an oracle backed by
  `elicit_logprobs`-style batched comparisons over arbitrary pair lists (add a small helper
  `compare_pairs(tok, model, items, pairs)` in `elicit.py` that returns `{(i,j): P(pick i)}` for an
  explicit pair list — needed by the sorter oracle); then `spacing_pass` + `fit_thurstone_sparse`.
  Reports Spearman ρ and MAE between dense μ and sparse μ, and comparison counts (dense vs efficient)
  + ratio. Writes JSON to a run folder.

- [ ] **Step 2:** Add `compare_pairs(tok, model, items, pairs, batch_size=64)` to `elicit.py`
  (batched forward over the given explicit ordered pairs, same A/B-logprob logic as
  `elicit_logprobs`). Refactor `elicit_logprobs` to call it over all i≠j pairs to avoid duplication.
- [ ] **Step 3: Sanity import** (CPU): `uv run python -c "import sentiment_utility.elicit"`.
- [ ] **Step 4: Commit:** `git add scripts/validate_method.py src/sentiment_utility/elicit.py && git commit -m "feat: dense-vs-efficient method validation + compare_pairs oracle"`

---

### Task 6: Activation extraction + full scale run (GPU)

**Files:** Create `scripts/run_scale.py`; add `extract_activations` to `probe.py`.

- [ ] **Step 1: Add `extract_activations(tok, model, items, batch_size=16) -> dict[int, np.ndarray]`**
  to `probe.py` (import torch lazily). For each batch: build neutral prompt per concept via the chat
  template (user content = just the concept string), tokenize left-padded, forward with
  `output_hidden_states=True`, take the **last-token** hidden state per layer → stack to
  (N, d) per layer. Return `{layer_index: array}` for all hidden layers.

- [ ] **Step 2: Implement `scripts/run_scale.py`**:
  - load items_500 + meta; load model once.
  - efficient elicit: `compare_pairs`-backed oracle → `rank_by_quicksort` → `spacing_pass` →
    `fit_thurstone_sparse`.
  - metrics on implied matrix (`predict_pref_matrix` → `cyclic_triad_fraction`,
    `expected_cycle_probability`, `completeness`); record `comparison_count` and ratio vs n(n-1).
  - probe: `extract_activations` → `probe_all_layers` targeting μ; if ≥50 items have human_valence,
    also probe valence; correlate μ with human_valence where available.
  - plots (seaborn → PDF): top/bottom-25 sentiment bar chart, probe R²-vs-layer, pairwise-acc-vs-layer.
  - timestamped `runs/<ts>/` with config + commit hash + edges + μ/σ + metrics + probe results + plots.

- [ ] **Step 3: Sanity import** (CPU): `uv run python -c "import sentiment_utility.probe"`.
- [ ] **Step 4: Commit:** `git add scripts/run_scale.py src/sentiment_utility/probe.py && git commit -m "feat: activation extraction + full 500-concept scale run"`

---

### Task 7: Full suite

- [ ] **Step 1:** `uv run pytest -v` — all tests (old + new) pass on CPU.
- [ ] **Step 2:** Update `README.md` with the new scripts (build_dataset, validate_method, run_scale)
  and the O(n log n) method note.
- [ ] **Step 3: Commit:** `git add -A && git commit -m "docs: README for scale/efficient/probe workflow"`

---

## Self-Review Notes
- **Spec coverage:** dataset blend+sample (T1/T4), O(n log n) sorter + sparse fit (T2), method
  validation (T5), probe + activations (T3/T6), full run + plots (T6), scikit-learn dep (T0). ✓
- **Type consistency:** `oracle(pairs:list[(i,j)]) -> {(i,j):p}` used identically in efficient.py,
  tests, and the `compare_pairs` GPU backing (T5). `fit_thurstone_sparse` returns the same key set as
  `fit_thurstone` plus `comparison_count`. `probe_all_layers` hidden dict keyed by layer index. ✓
- **GPU vs CPU:** sampler, sorter, sparse fit, probe training all CPU-testable via synthetic
  oracles/activations; only `compare_pairs`, `extract_activations`, and the scripts need GPU. ✓
- **Dependency on `_phi`:** efficient.py imports `_phi` from thurstone.py — implementer must confirm
  it's exposed (it is defined at module scope there). ✓
