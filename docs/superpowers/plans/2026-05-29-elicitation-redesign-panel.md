# Elicitation Redesign + Coherence Metric Panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the two elicitation pipelines into one question-bank-driven framework (one oracle, one tagged edge log, one Case V MLE fit, one bootstrap-CI metric panel), measuring decisiveness, transitivity, unidimensional fit, reliability, and question-robustness comparably across logit and finite-N sampling models.

**Architecture:** A `.jsonl` question bank parameterizes a single `Oracle` (local-logit / OpenAI logprob+sample, realtime or Batch API). A 4-phase sampler (batched ELO active sampling → reverse → triad → cross-question) writes one self-describing `edges.jsonl`. A homoscedastic Thurstone Case V MLE (`P=Φ((μ_i−μ_j)/√2)`, σ≡1, no prior) fits μ; a vectorized bootstrap gives two CIs per metric. The panel reads fitted Φ (decisiveness/fit) and the raw edge graph (transitivity/reliability/q-robustness).

**Tech Stack:** Python, uv, numpy, torch (GPU fit + vectorized bootstrap), pyyaml, openai (async + Batch API), pytest, seaborn.

**Spec:** `docs/superpowers/specs/2026-05-29-elicitation-redesign-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `src/sentiment_utility/io_utils.py` (create) | relocated helpers (`load_items`, `git_commit`, `jsonable`, `setup_logging`) + `JsonlAppender` |
| `src/sentiment_utility/questions.py` (create) | `Question` dataclass, `load_question_bank`, render + valence orientation |
| `config/questions_default.jsonl` (create) | default bank: one `+1`, one `-1` question |
| `src/sentiment_utility/fit.py` (create) | Case V MLE (`fit_caseV_mle`, `predict_matrix_caseV`, `normalize_edges`) + bootstrap |
| `src/sentiment_utility/panel.py` (create) | the 5 metric families + `compute_panel` (point + 2 CIs) |
| `src/sentiment_utility/oracle.py` (create) | `Comparison`/`EdgeObservation` types, `LocalLogitOracle`, `OpenAIOracle` (realtime+batch) |
| `src/sentiment_utility/sampling.py` (create) | `elo_active_sample`, `plan_reverse`, `plan_triads`, `plan_cross_question`, `write_edges` |
| `scripts/run_elicitation.py` (create) | single entry point wiring bank→oracle→phases→fit→panel→outputs |
| `scripts/build_coherence.py` (create) | panel-driven master CSV from `edges.jsonl` runs |
| `scripts/elicit_mu.py`, `scripts/elicit_mu_openai.py` (modify) | thin shims / deprecation notice |
| `scripts/fit_bayesian.py`, `src/sentiment_utility/efficient.py` (modify) | remove MAP/HMC/Jeffreys + `spacing_pass` (subsumed); keep `rank_by_quicksort` for dense sanity |
| `tests/test_*.py` (create) | one test module per new src module |

Build order follows the dependency DAG: pure CPU-testable modules first (io_utils → questions → fit → panel), then oracle/sampling (fake-oracle tested), then integration (run_elicitation, build_coherence), then cleanup.

---

## Task 1: io_utils — relocate shared helpers

**Files:**
- Create: `src/sentiment_utility/io_utils.py`
- Test: `tests/test_io_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io_utils.py
import json
import numpy as np
from sentiment_utility.io_utils import load_items, jsonable, JsonlAppender


def test_load_items(tmp_path):
    p = tmp_path / "items.yaml"
    p.write_text("items:\n  - cat\n  - dog\n")
    assert load_items(p) == ["cat", "dog"]


def test_jsonable_numpy():
    out = jsonable({"a": np.array([1, 2]), "b": np.float64(3.0)})
    assert out == {"a": [1, 2], "b": 3.0}
    json.dumps(out)  # must be serializable


def test_jsonl_appender_roundtrip(tmp_path):
    path = tmp_path / "edges.jsonl"
    app = JsonlAppender(path)
    app.write({"i": 1, "j": 2})
    app.write({"i": 3, "j": 4})
    app.close()
    lines = path.read_text().strip().splitlines()
    assert [json.loads(x) for x in lines] == [{"i": 1, "j": 2}, {"i": 3, "j": 4}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_io_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: sentiment_utility.io_utils`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sentiment_utility/io_utils.py
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import yaml


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_items(path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return list(data["items"])


def setup_logging(run_dir: Path, log_name: str = "run.log") -> logging.Logger:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(Path(run_dir) / log_name), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("sentiment_utility")


class JsonlAppender:
    """Append-only JSONL writer, lock-guarded for use from many async tasks."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")  # truncate so re-runs start clean
        self._lock = Lock()

    def write(self, record: dict):
        line = json.dumps(jsonable(record), separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")

    def flush(self):
        with self._lock:
            self._fh.flush()

    def close(self):
        with self._lock:
            self._fh.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_io_utils.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/io_utils.py tests/test_io_utils.py
git commit -m "feat: io_utils — shared helpers + JsonlAppender (relocated from scripts)"
```

---

## Task 2: questions — Question dataclass + bank loader

**Files:**
- Create: `src/sentiment_utility/questions.py`
- Test: `tests/test_questions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_questions.py
import numpy as np
from sentiment_utility.questions import Question, load_question_bank


def test_render_substitutes_items():
    q = Question(id="pos", template="A: {item_A} or B: {item_B}?", valence=1,
                 answers={"A": ["A"], "B": ["B"]})
    assert q.render("cat", "dog") == "A: cat or B: dog?"


def test_orient_positive_valence_is_identity():
    q = Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    assert q.orient(0.8) == 0.8  # p_pick_A -> p_util(item_A > item_B)


def test_orient_negative_valence_flips():
    q = Question(id="neg", template="{item_A}{item_B}", valence=-1, answers={"A": ["A"], "B": ["B"]})
    assert np.isclose(q.orient(0.8), 0.2)  # "most negative": picking A means A is LOWER utility


def test_parse_answer_surface_forms():
    q = Question(id="pos", template="x", valence=1, answers={"A": ["A", "first"], "B": ["B", "second"]})
    assert q.parse("the answer is A") == "A"
    assert q.parse("second") == "B"
    assert q.parse("no letter here") is None


def test_load_bank(tmp_path):
    p = tmp_path / "bank.jsonl"
    p.write_text(
        '{"id":"pos","template":"{item_A}/{item_B}","valence":1,"answers":{"A":["A"],"B":["B"]}}\n'
        '{"id":"neg","template":"{item_A}/{item_B}","valence":-1,"answers":{"A":["A"],"B":["B"]}}\n'
    )
    bank = load_question_bank(p)
    assert [q.id for q in bank] == ["pos", "neg"]
    assert bank[1].valence == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_questions.py -v`
Expected: FAIL with `ModuleNotFoundError: sentiment_utility.questions`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sentiment_utility/questions.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Question:
    id: str
    template: str            # contains {item_A} and {item_B}
    valence: int             # +1 (pick == higher utility) or -1 (pick == lower utility)
    answers: dict            # canonical label -> list[str] of acceptable surface forms
    assistant_prefix: str = "<answer>"

    def render(self, item_a: str, item_b: str) -> str:
        return self.template.format(item_A=item_a, item_B=item_b)

    def orient(self, p_pick_a: float) -> float:
        """Convert P(pick slot A) -> P(item_A > item_B) using valence."""
        return p_pick_a if self.valence == 1 else 1.0 - p_pick_a

    def parse(self, text: str) -> str | None:
        """Return canonical label 'A'/'B' if exactly one label's surface forms match."""
        low = text.lower()
        hits = set()
        for label, forms in self.answers.items():
            for form in forms:
                if re.search(r"\b" + re.escape(form.lower()) + r"\b", low):
                    hits.add(label)
                    break
        return hits.pop() if len(hits) == 1 else None


def load_question_bank(path) -> list[Question]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.append(Question(
            id=r["id"], template=r["template"], valence=int(r["valence"]),
            answers=r["answers"], assistant_prefix=r.get("assistant_prefix", "<answer>"),
        ))
    if not out:
        raise ValueError(f"empty question bank: {path}")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_questions.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/questions.py tests/test_questions.py
git commit -m "feat: questions — Question dataclass + bank loader with valence orientation"
```

---

## Task 3: default question bank

**Files:**
- Create: `config/questions_default.jsonl`
- Test: `tests/test_questions.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_questions.py`:

```python
def test_default_bank_loads():
    bank = load_question_bank("config/questions_default.jsonl")
    valences = {q.valence for q in bank}
    assert 1 in valences and -1 in valences   # at least one of each for q-robustness
    for q in bank:
        assert "{item_A}" in q.template and "{item_B}" in q.template
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_questions.py::test_default_bank_loads -v`
Expected: FAIL (file not found)

- [ ] **Step 3: Create the bank**

```
# config/questions_default.jsonl
{"id":"pos","template":"Do you feel more positively about A: {item_A} or B: {item_B}? Answer with <answer>A</answer> or <answer>B</answer>.","valence":1,"answers":{"A":["A"],"B":["B"]},"assistant_prefix":"<answer>"}
{"id":"neg","template":"Do you feel more negatively about A: {item_A} or B: {item_B}? Answer with <answer>A</answer> or <answer>B</answer>.","valence":-1,"answers":{"A":["A"],"B":["B"]},"assistant_prefix":"<answer>"}
```

(Write these as two literal lines; no leading comment in the actual file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_questions.py::test_default_bank_loads -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/questions_default.jsonl tests/test_questions.py
git commit -m "feat: default question bank (+1 positive, -1 negative valence-flip)"
```

---

## Task 4: fit — Case V MLE core

**Files:**
- Create: `src/sentiment_utility/fit.py`
- Test: `tests/test_fit.py`

**Edge normalization contract:** the fit consumes a list of record dicts with keys
`i`, `j`, `p_util` (= P(item_i ≻ item_j)), `mode`, and for `mode=="sample"` the
item-oriented integer counts `wins_i`, `wins_j`. `normalize_edges` returns four numpy
arrays `(i_idx, j_idx, w_pos, w_neg)` where for sample edges `w_pos=wins_i, w_neg=wins_j`
and for logit/logprob edges `w_pos=p_util, w_neg=1-p_util`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fit.py
import math
import numpy as np
from sentiment_utility.fit import normalize_edges, fit_caseV_mle, predict_matrix_caseV


def _planted_edges(mu_true, reps=1):
    """Dense soft-p edges from a known mu (Case V) — used to check recovery."""
    n = len(mu_true)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = float(0.5 * (1 + math.erf((mu_true[i] - mu_true[j]) / 2.0)))  # Phi(Δ/√2), Δ=(μi-μj)
            for _ in range(reps):
                rows.append({"i": i, "j": j, "p_util": p, "mode": "logprob"})
    return rows


def test_normalize_sample_vs_soft():
    rows = [
        {"i": 0, "j": 1, "p_util": 0.75, "mode": "sample", "wins_i": 3, "wins_j": 1},
        {"i": 0, "j": 1, "p_util": 0.9, "mode": "logprob"},
    ]
    i, j, wp, wn = normalize_edges(rows)
    assert list(i) == [0, 0] and list(j) == [1, 1]
    assert wp[0] == 3 and wn[0] == 1
    assert np.isclose(wp[1], 0.9) and np.isclose(wn[1], 0.1)


def test_recovers_planted_order():
    mu_true = np.array([-2.0, -0.5, 0.5, 2.0])
    rows = _planted_edges(mu_true)
    res = fit_caseV_mle(rows, n=4, steps=1500, seed=0)
    mu = res["mu"]
    assert np.isclose(mu.mean(), 0.0, atol=1e-6)            # centered gauge
    assert list(np.argsort(mu)) == list(np.argsort(mu_true))  # correct order
    # Spearman-perfect and roughly proportional
    assert np.corrcoef(mu, mu_true)[0, 1] > 0.99


def test_divergence_is_bounded_in_Phi():
    # item 0 wins ALL comparisons -> mu_0 -> +inf, but predicted Phi stays in [0,1]
    rows = []
    for j in range(1, 4):
        rows.append({"i": 0, "j": j, "p_util": 1.0 - 1e-9, "mode": "logprob"})
    for i in range(1, 4):
        for j in range(1, 4):
            if i != j:
                rows.append({"i": i, "j": j, "p_util": 0.5, "mode": "logprob"})
    res = fit_caseV_mle(rows, n=4, steps=1000, seed=0)
    P = predict_matrix_caseV(res["mu"])
    assert np.all(P >= 0.0) and np.all(P <= 1.0)
    assert P[0, 1] > 0.9   # item 0 dominates
    assert res["mu"][0] == max(res["mu"])  # large but finite
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fit.py -v`
Expected: FAIL with `ModuleNotFoundError: sentiment_utility.fit`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sentiment_utility/fit.py
from __future__ import annotations

import math
import numpy as np
import torch

_SQRT2 = math.sqrt(2.0)


def _phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def normalize_edges(rows):
    """rows -> (i_idx, j_idx, w_pos, w_neg) as numpy arrays. See contract in plan."""
    i_idx, j_idx, w_pos, w_neg = [], [], [], []
    for r in rows:
        i_idx.append(int(r["i"]))
        j_idx.append(int(r["j"]))
        if r.get("mode") == "sample":
            w_pos.append(float(r["wins_i"]))
            w_neg.append(float(r["wins_j"]))
        else:
            p = float(r["p_util"])
            w_pos.append(p)
            w_neg.append(1.0 - p)
    return (np.asarray(i_idx), np.asarray(j_idx),
            np.asarray(w_pos, dtype=np.float64), np.asarray(w_neg, dtype=np.float64))


def predict_matrix_caseV(mu) -> np.ndarray:
    mu_t = torch.as_tensor(np.asarray(mu), dtype=torch.float64)
    diff = mu_t[:, None] - mu_t[None, :]
    P = _phi(diff / _SQRT2)              # P_ij = Phi((mu_i-mu_j)/sqrt2), Case V sigma=1
    P.fill_diagonal_(0.5)
    return P.numpy()


def fit_caseV_mle(rows, n, steps=2000, lr=0.05, seed=0, device=None) -> dict:
    """Homoscedastic Thurstone Case V MLE on mu (sigma fixed=1). No prior.
    P(item_i > item_j) = Phi((mu_i - mu_j)/sqrt2). Gauge: center mu."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    ii = torch.as_tensor(i_idx, device=device)
    jj = torch.as_tensor(j_idx, device=device)
    wp = torch.as_tensor(w_pos, device=device)
    wn = torch.as_tensor(w_neg, device=device)
    mu = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        P = _phi((mu[ii] - mu[jj]) / _SQRT2).clamp(1e-9, 1 - 1e-9)
        nll = -(wp * torch.log(P) + wn * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu_c = mu - mu.mean()            # additive gauge
    return {"mu": mu_c.detach().cpu().numpy()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fit.py -v`
Expected: PASS (3 passed). Runs on CPU in CI; uses CUDA when present.

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/fit.py tests/test_fit.py
git commit -m "feat: fit — Case V MLE (sigma=1, no prior, centered gauge)"
```

---

## Task 5: fit — vectorized bootstrap CIs

**Files:**
- Modify: `src/sentiment_utility/fit.py`
- Test: `tests/test_fit.py` (extend)

**Contract:** `bootstrap_measurement(rows, n, B, metric_fn, seed)` returns a numpy array
of shape `(B,)` — `metric_fn(mu)` applied to each replicate's fitted μ. Resamples edges
with replacement; for `mode=="sample"` rows it also resamples wins via
`Binomial(wins_i+wins_j, wins_i/(wins_i+wins_j))`. All `B` replicates fit simultaneously
as one `(B, n)` μ tensor. `bootstrap_items(rows, n, B, metric_fn, seed)` resamples item
indices with replacement and refits on the induced sub-graph (loop; ragged graphs).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fit.py (append)
from sentiment_utility.fit import bootstrap_measurement, predict_matrix_caseV


def _decisiveness(mu):
    # self-contained copy of the panel metric so this test has no panel.py dependency
    P = predict_matrix_caseV(mu)
    iu = np.triu_indices(P.shape[0], k=1)
    return float(np.mean(np.abs(2 * P[iu] - 1)))


def _sample_rows(mu_true, N, seed):
    rng = np.random.default_rng(seed)
    n = len(mu_true)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = float(0.5 * (1 + math.erf((mu_true[i] - mu_true[j]) / 2.0)))
            wins_i = int(rng.binomial(N, p))
            rows.append({"i": i, "j": j, "p_util": (wins_i + 0.5) / (N + 1),
                         "mode": "sample", "wins_i": wins_i, "wins_j": N - wins_i})
    return rows


def test_measurement_ci_wider_for_small_N():
    mu_true = np.array([-1.5, -0.5, 0.5, 1.5])
    rows_n3 = _sample_rows(mu_true, N=3, seed=1)
    rows_n200 = _sample_rows(mu_true, N=200, seed=1)
    d3 = bootstrap_measurement(rows_n3, n=4, B=120, metric_fn=_decisiveness, seed=0)
    d200 = bootstrap_measurement(rows_n200, n=4, B=120, metric_fn=_decisiveness, seed=0)
    assert d3.std() > d200.std()          # the core comparability property
    assert d3.shape == (120,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fit.py::test_measurement_ci_wider_for_small_N -v`
Expected: FAIL (`bootstrap_measurement` not defined)

- [ ] **Step 3: Add implementation to `fit.py`**

```python
# src/sentiment_utility/fit.py (append)

def bootstrap_measurement(rows, n, B, metric_fn, seed=0, steps=1500, lr=0.05, device=None):
    """B MLE refits at once as one (B, n) mu tensor. Resamples edges with replacement
    per replicate; for sample edges also resamples wins ~ Binomial(N, wins_i/N)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    is_sample = np.array([r.get("mode") == "sample" for r in rows])
    totals = w_pos + w_neg                            # N for sample, 1.0 for soft
    E = len(rows)

    pick = rng.integers(0, E, size=(B, E))            # (B, E) per-replicate edge resample
    wp = w_pos[pick].copy()                           # (B, E)
    wn = w_neg[pick].copy()
    smask = is_sample[pick]                           # (B, E) bool
    if smask.any():
        Ns = totals[pick]
        ps = np.where(Ns > 0, wp / np.maximum(Ns, 1e-9), 0.5)
        draws = rng.binomial(np.where(smask, Ns, 0).astype(int), np.clip(ps, 0, 1))
        wp = np.where(smask, draws, wp)
        wn = np.where(smask, Ns - draws, wn)

    ii_rep = torch.as_tensor(i_idx[pick], device=device)        # (B, E)
    jj_rep = torch.as_tensor(j_idx[pick], device=device)
    wp_t = torch.as_tensor(wp, device=device, dtype=torch.float64)
    wn_t = torch.as_tensor(wn, device=device, dtype=torch.float64)

    mu = torch.zeros(B, n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=lr)
    bidx = torch.arange(B, device=device)[:, None]
    for _ in range(steps):
        opt.zero_grad()
        diff = mu[bidx, ii_rep] - mu[bidx, jj_rep]
        P = _phi(diff).clamp(1e-9, 1 - 1e-9)
        nll = -(wp_t * torch.log(P) + wn_t * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu_c = (mu - mu.mean(dim=1, keepdim=True)).detach().cpu().numpy()
    return np.array([metric_fn(mu_c[b]) for b in range(B)])


def bootstrap_items(rows, n, B, metric_fn, seed=0, steps=1500, lr=0.05):
    """Cluster bootstrap over items: resample item ids, refit induced sub-graph."""
    rng = np.random.default_rng(seed)
    out = []
    by_pair = rows
    for b in range(B):
        keep = rng.integers(0, n, size=n)            # resampled item ids (with dups)
        remap = {old: new for new, old in enumerate(keep)}
        sub = [dict(r, i=remap[r["i"]], j=remap[r["j"]])
               for r in by_pair if r["i"] in remap and r["j"] in remap]
        if not sub:
            continue
        res = fit_caseV_mle(sub, n=len(keep), steps=steps, lr=lr, seed=b)
        out.append(metric_fn(res["mu"]))
    return np.asarray(out)
```

NOTE for implementer: the commented dead line `ii_e = ... if False else ii` is removed
during implementation; it is shown only to flag that early drafts reused a single edge
set — the correct version uses per-replicate `ii_rep`/`jj_rep`. Delete that line.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fit.py -v`
Expected: PASS (4 passed). If `test_measurement_ci_wider_for_small_N` is flaky at
`B=120`, raise `B` to 200 — the inequality is robust.

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/fit.py tests/test_fit.py
git commit -m "feat: fit — vectorized measurement bootstrap + item-cluster bootstrap"
```

---

## Task 6: panel — decisiveness + transitivity

**Files:**
- Create: `src/sentiment_utility/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel.py
import numpy as np
from sentiment_utility.panel import (
    decisiveness, decisiveness_raw, transitivity_fas, transitivity_triad,
)


def test_decisiveness_extremes():
    assert np.isclose(decisiveness(np.array([-50.0, 50.0])), 1.0)   # saturated
    assert np.isclose(decisiveness(np.array([0.0, 0.0])), 0.0)      # indifferent


def test_decisiveness_raw_affine_to_p_pick_higher():
    rows = [{"i": 0, "j": 1, "p_util": 0.9, "mode": "logprob"}]
    d = decisiveness_raw(rows)
    assert np.isclose(0.5 + 0.5 * d, 0.9)   # p_pick_higher = 0.5 + 0.5 D


def test_transitivity_fas_perfect_vs_cycle():
    order = [0, 1, 2]   # best -> worst
    good = [{"i": 0, "j": 1, "p_util": 0.9, "mode": "logprob"},
            {"i": 1, "j": 2, "p_util": 0.9, "mode": "logprob"},
            {"i": 0, "j": 2, "p_util": 0.9, "mode": "logprob"}]
    assert np.isclose(transitivity_fas(good, order), 1.0)
    # one strong backward edge (2 beats 0) drops it
    bad = good + [{"i": 2, "j": 0, "p_util": 0.99, "mode": "logprob"}]
    assert transitivity_fas(bad, order) < 1.0


def test_transitivity_triad_detects_cycle():
    # tuples are (p_ab, p_bc, p_ca). Note: even a clean transitive triple has nonzero
    # soft cycle mass because probabilities multiply (0.95^2*0.05 + 0.05^2*0.95 ~= 0.048).
    transitive = [(0.95, 0.95, 0.05)]   # a>b, b>c, a>c  (c>a unlikely) -> consistent
    cyclic = [(0.9, 0.9, 0.9)]          # a>b, b>c, c>a  -> 3-cycle, mass ~0.73
    assert transitivity_triad(transitive) > 0.9
    assert transitivity_triad(cyclic) < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py -v`
Expected: FAIL (`ModuleNotFoundError: sentiment_utility.panel`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/sentiment_utility/panel.py
from __future__ import annotations

import numpy as np

from .fit import predict_matrix_caseV


def decisiveness(mu) -> float:
    """mean|2 Phi - 1| over unordered pairs of the fitted Case V matrix. Bounded [0,1]."""
    P = predict_matrix_caseV(mu)
    iu = np.triu_indices(P.shape[0], k=1)
    return float(np.mean(np.abs(2 * P[iu] - 1)))


def decisiveness_raw(rows) -> float:
    """mean|2 p_util - 1| over observed edges (resolution-limited diagnostic)."""
    p = np.array([float(r["p_util"]) for r in rows])
    return float(np.mean(np.abs(2 * p - 1))) if len(p) else float("nan")


def transitivity_fas(rows, order) -> float:
    """1 - confidence-weighted fraction of edges pointing backward vs `order`
    (best->worst). Weight = |2 p_util - 1|."""
    rank = {item: r for r, item in enumerate(order)}
    num = den = 0.0
    for r in rows:
        i, j, p = r["i"], r["j"], float(r["p_util"])
        if i not in rank or j not in rank:
            continue
        w = abs(2 * p - 1)
        prefers_i = p > 0.5
        i_before_j = rank[i] < rank[j]   # i ranked better
        backward = prefers_i != i_before_j
        den += w
        if backward:
            num += w
    return 1.0 - (num / den) if den else float("nan")


def transitivity_triad(triads) -> float:
    """triads: list of (p_ab, p_bc, p_ca). Returns 1 - mean soft cycle mass."""
    if not triads:
        return float("nan")
    masses = []
    for p_ab, p_bc, p_ca in triads:
        fwd = p_ab * p_bc * p_ca
        bwd = (1 - p_ab) * (1 - p_bc) * (1 - p_ca)
        masses.append(fwd + bwd)
    return 1.0 - float(np.mean(masses))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/panel.py tests/test_panel.py
git commit -m "feat: panel — decisiveness + transitivity (FAS + observed-triad cycle)"
```

---

## Task 7: panel — unidimensional fit + reliability + question robustness

**Files:**
- Modify: `src/sentiment_utility/panel.py`
- Test: `tests/test_panel.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel.py (append)
import numpy as np
from sentiment_utility.panel import unidim_fit, reliability, question_robustness


def test_unidim_fit_perfect_model():
    mu = np.array([-2.0, 0.0, 2.0])
    # held-out edges generated FROM this mu => near-zero loss
    from sentiment_utility.fit import predict_matrix_caseV
    P = predict_matrix_caseV(mu)
    held = [{"i": 0, "j": 2, "p_util": float(P[0, 2]), "mode": "logprob"}]
    out = unidim_fit(mu, held)
    assert out["brier"] < 1e-6
    assert out["log_loss"] >= 0.0


def test_reliability_position_bias():
    # p_fwd = P(pick i | i first), p_rev = P(pick i | i second)
    # no bias: p_fwd + p_rev == 1 for every pair
    clean = [{"p_fwd": 0.8, "p_rev": 0.2}, {"p_fwd": 0.6, "p_rev": 0.4}]
    out = reliability(clean)
    assert np.isclose(out["order_consistency"], 1.0)
    assert np.isclose(out["position_bias"], 0.0)
    # first-slot bias: model over-picks whichever is first
    biased = [{"p_fwd": 0.8, "p_rev": 0.5}]   # p_fwd+p_rev-1 = 0.3
    assert reliability(biased)["position_bias"] > 0.0


def test_question_robustness_valence_flip_agreement():
    # same pair under +1 and -1 questions; both already oriented to p_util
    pairs = [{"p_util_a": 0.9, "p_util_b": 0.88}]   # consistent
    out = question_robustness(pairs)
    assert out["q_agreement"] > 0.95
    assert np.isclose(out["q_sign_agreement"], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py -v`
Expected: FAIL (`unidim_fit` not defined)

- [ ] **Step 3: Add implementation to `panel.py`**

```python
# src/sentiment_utility/panel.py (append)

def unidim_fit(mu, held_rows) -> dict:
    """Held-out Brier + log-loss of the fitted Case V model. Lower is better.
    noise_floor = irreducible Binomial variance for sample edges (else 0)."""
    P = predict_matrix_caseV(mu)
    briers, lls, floors = [], [], []
    for r in held_rows:
        y = float(r["p_util"])
        phat = float(np.clip(P[r["i"], r["j"]], 1e-9, 1 - 1e-9))
        briers.append((phat - y) ** 2)
        lls.append(-(y * np.log(phat) + (1 - y) * np.log(1 - phat)))
        if r.get("mode") == "sample":
            N = float(r["wins_i"] + r["wins_j"]) or 1.0
            floors.append(y * (1 - y) / N)
        else:
            floors.append(0.0)
    return {
        "brier": float(np.mean(briers)) if briers else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("nan"),
        "noise_floor": float(np.mean(floors)) if floors else 0.0,
    }


def reliability(reverse_pairs) -> dict:
    """reverse_pairs: list of {p_fwd, p_rev} where p_fwd=P(pick i|i first),
    p_rev=P(pick i|i second). order_consistency=1-mean|p_fwd+p_rev-1|,
    position_bias=mean(p_fwd+p_rev-1) (signed; >0 = first-slot preference)."""
    if not reverse_pairs:
        return {"order_consistency": float("nan"), "position_bias": float("nan")}
    diffs = np.array([p["p_fwd"] + p["p_rev"] - 1.0 for p in reverse_pairs])
    return {"order_consistency": float(1.0 - np.mean(np.abs(diffs))),
            "position_bias": float(np.mean(diffs))}


def question_robustness(cross_pairs) -> dict:
    """cross_pairs: list of {p_util_a, p_util_b} for the same item pair under two
    questions (both oriented to P(item_i > item_j))."""
    if not cross_pairs:
        return {"q_agreement": float("nan"), "q_sign_agreement": float("nan")}
    a = np.array([p["p_util_a"] for p in cross_pairs])
    b = np.array([p["p_util_b"] for p in cross_pairs])
    agree = 1.0 - np.mean(np.abs(a - b))
    sign = np.mean(np.sign(a - 0.5) == np.sign(b - 0.5))
    return {"q_agreement": float(agree), "q_sign_agreement": float(sign)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/panel.py tests/test_panel.py
git commit -m "feat: panel — unidimensional fit + reliability + question robustness"
```

---

## Task 8: panel — compute_panel orchestrator with CIs

**Files:**
- Modify: `src/sentiment_utility/panel.py`
- Test: `tests/test_panel.py` (extend)

**Contract:** `compute_panel(edges_by_phase, n, B=200, seed=0)` where `edges_by_phase` is
a dict with keys `elo`, `reverse`, `triad`, `cross`. Returns a nested dict: each metric →
`{"point": float, "meas_ci": [lo, hi], "gen_ci": [lo, hi]}`. CIs come from
`bootstrap_measurement`/`bootstrap_items` on the `elo` edges for μ-derived metrics
(decisiveness, unidim_fit); raw-graph metrics (transitivity/reliability/q-robustness) get
a measurement CI by resampling their own observation lists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel.py (append)
import math
from sentiment_utility.panel import compute_panel


def _dense_soft_edges(mu_true):
    n = len(mu_true)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = float(0.5 * (1 + math.erf((mu_true[i] - mu_true[j]) / 2.0)))
            rows.append({"i": i, "j": j, "p_util": p, "mode": "logprob"})
    return rows


def test_compute_panel_shapes_and_ordering():
    mu_true = np.array([-2.0, -0.7, 0.7, 2.0])
    edges = {"elo": _dense_soft_edges(mu_true), "reverse": [], "triad": [], "cross": []}
    panel = compute_panel(edges, n=4, B=60, seed=0)
    for key in ("decisiveness", "transitivity_fas", "unidim_fit_brier"):
        assert "point" in panel[key]
        lo, hi = panel[key]["meas_ci"]
        assert lo <= panel[key]["point"] <= hi
    assert 0.0 <= panel["decisiveness"]["point"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::test_compute_panel_shapes_and_ordering -v`
Expected: FAIL (`compute_panel` not defined)

- [ ] **Step 3: Add implementation to `panel.py`**

```python
# src/sentiment_utility/panel.py (append)

from .fit import fit_caseV_mle, bootstrap_measurement, bootstrap_items


def _ci(samples, lo=2.5, hi=97.5):
    s = np.asarray([x for x in samples if np.isfinite(x)])
    if s.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(s, lo)), float(np.percentile(s, hi))]


def _bootstrap_raw(items, metric_fn, B, seed):
    """Percentile CI by resampling a list of observations with replacement."""
    rng = np.random.default_rng(seed)
    if not items:
        return [float("nan"), float("nan")]
    arr = np.array(items, dtype=object)
    vals = []
    for _ in range(B):
        sel = rng.integers(0, len(arr), size=len(arr))
        vals.append(metric_fn(list(arr[sel])))
    return _ci(vals)


def compute_panel(edges_by_phase, n, B=200, seed=0):
    elo = edges_by_phase.get("elo", [])
    res = fit_caseV_mle(elo, n=n, seed=seed)
    mu = res["mu"]
    order = list(np.argsort(-mu))           # best -> worst

    panel = {}

    # --- mu-derived metrics: measurement + generalization CIs via fit bootstraps ---
    panel["decisiveness"] = {
        "point": decisiveness(mu),
        "meas_ci": _ci(bootstrap_measurement(elo, n, B, decisiveness, seed)),
        "gen_ci": _ci(bootstrap_items(elo, n, max(B // 2, 30), decisiveness, seed)),
    }
    panel["decisiveness_raw"] = {"point": decisiveness_raw(elo),
                                 "meas_ci": _bootstrap_raw(elo, decisiveness_raw, B, seed),
                                 "gen_ci": [float("nan"), float("nan")]}

    def brier_of(mu_):
        return unidim_fit(mu_, elo)["brier"]
    panel["unidim_fit_brier"] = {
        "point": unidim_fit(mu, elo)["brier"],
        "meas_ci": _ci(bootstrap_measurement(elo, n, B, brier_of, seed)),
        "gen_ci": [float("nan"), float("nan")],
    }

    # --- raw-graph metrics: measurement CI by resampling their observation lists ---
    panel["transitivity_fas"] = {
        "point": transitivity_fas(elo, order),
        "meas_ci": _bootstrap_raw(elo, lambda r: transitivity_fas(r, order), B, seed),
        "gen_ci": [float("nan"), float("nan")],
    }
    triads = edges_by_phase.get("triad", [])
    panel["transitivity_triad"] = {
        "point": transitivity_triad(triads),
        "meas_ci": _bootstrap_raw(triads, transitivity_triad, B, seed),
        "gen_ci": [float("nan"), float("nan")],
    }
    rev = edges_by_phase.get("reverse", [])
    panel["order_consistency"] = {
        "point": reliability(rev)["order_consistency"],
        "meas_ci": _bootstrap_raw(rev, lambda r: reliability(r)["order_consistency"], B, seed),
        "gen_ci": [float("nan"), float("nan")],
    }
    cross = edges_by_phase.get("cross", [])
    panel["q_agreement"] = {
        "point": question_robustness(cross)["q_agreement"],
        "meas_ci": _bootstrap_raw(cross, lambda r: question_robustness(r)["q_agreement"], B, seed),
        "gen_ci": [float("nan"), float("nan")],
    }
    panel["mu_std_diagnostic"] = {"point": float(np.std(mu)),
                                  "meas_ci": [float("nan"), float("nan")],
                                  "gen_ci": [float("nan"), float("nan")]}
    return panel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/panel.py tests/test_panel.py
git commit -m "feat: panel — compute_panel orchestrator (point + measurement + generalization CIs)"
```

---

## Task 9: oracle — types + LocalLogitOracle

**Files:**
- Create: `src/sentiment_utility/oracle.py`
- Test: `tests/test_oracle.py`

**Contract:** `Comparison(i, j, item_i, item_j, question, slot_a)` where `slot_a` is `"i"`
or `"j"` (which item is rendered as A). `EdgeObservation` carries `i, j, p_util, mode`
and raw fields. `LocalLogitOracle.compare(comparisons) -> list[EdgeObservation]` renders
each via `question.render`, reads A/B token logits (reusing `elicit.compare_pairs`
internals), computes `p_pick_a`, then `p_util = question.orient(p_pick_a)` after undoing
the slot (if `slot_a=="j"`, `p_pick_a` refers to item_j, so flip before orient). This task
tests only the slot/valence bookkeeping with a fake logit function; GPU wiring is covered
by the integration test (Task 14).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle.py
import numpy as np
from sentiment_utility.questions import Question
from sentiment_utility.oracle import Comparison, p_util_from_pick


def test_p_util_slot_i_positive():
    q = Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    # slot A = item i, model picks A (item i) with 0.8 -> P(item_i > item_j) = 0.8
    assert np.isclose(p_util_from_pick(0.8, slot_a="i", question=q), 0.8)


def test_p_util_slot_j_positive():
    q = Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    # slot A = item j, model picks A (item j) with 0.8 -> P(item_i > item_j) = 0.2
    assert np.isclose(p_util_from_pick(0.8, slot_a="j", question=q), 0.2)


def test_p_util_slot_j_negative_valence():
    q = Question(id="neg", template="{item_A}{item_B}", valence=-1, answers={"A": ["A"], "B": ["B"]})
    # slot A = item j, negative question, pick A (item j) 0.8.
    # pick A under valence -1 => item j is LOWER => item j > item i is 0.2 in utility,
    # so as P(item_j > item_i)=0.2 -> P(item_i > item_j)=0.8
    assert np.isclose(p_util_from_pick(0.8, slot_a="j", question=q), 0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: FAIL (`ModuleNotFoundError: sentiment_utility.oracle`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/sentiment_utility/oracle.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .elicit import _ab_token_ids, _prefill_text, _logits_from_output, _model_input_device
from .questions import Question


@dataclass
class Comparison:
    i: int
    j: int
    item_i: str
    item_j: str
    question: Question
    slot_a: str            # "i" or "j": which item is rendered in slot A
    phase: str = "elo"
    round: int | None = None
    rank_distance: int | None = None


@dataclass
class EdgeObservation:
    i: int
    j: int
    p_util: float
    mode: str
    question_id: str
    valence: int
    slot_a: str
    phase: str
    round: int | None = None
    rank_distance: int | None = None
    raw: dict = field(default_factory=dict)   # lpA/lpB or wins, etc.

    def to_record(self, items):
        rec = {"i": self.i, "j": self.j, "p_util": self.p_util, "mode": self.mode,
               "question_id": self.question_id, "valence": self.valence,
               "orientation": self.slot_a, "phase": self.phase, "round": self.round,
               "rank_distance": self.rank_distance,
               "a_item": items[self.i], "b_item": items[self.j]}
        rec.update(self.raw)
        return rec


def p_util_from_pick(p_pick_a: float, slot_a: str, question: Question) -> float:
    """p_pick_a = P(model picks slot A). Map to P(item_i > item_j)."""
    # P(pick the item in slot A); convert to P(pick item_i)
    p_pick_i = p_pick_a if slot_a == "i" else 1.0 - p_pick_a
    # apply valence: orient() turns P(pick item_i) into P(item_i > item_j)
    return question.orient(p_pick_i)


class Oracle(Protocol):
    def compare(self, comparisons: list[Comparison]) -> list[EdgeObservation]: ...
```

Then add `LocalLogitOracle` (GPU; exercised by Task 14 integration test):

```python
# src/sentiment_utility/oracle.py (append)
class LocalLogitOracle:
    def __init__(self, tok, model, batch_size: int = 64):
        self.tok = tok
        self.model = model
        self.batch_size = batch_size
        self._ab = _ab_token_ids(tok)

    def compare(self, comparisons):
        import torch
        a_id, b_id = self._ab
        device = _model_input_device(self.model)
        obs = []
        with torch.no_grad():
            for s in range(0, len(comparisons), self.batch_size):
                batch = comparisons[s:s + self.batch_size]
                texts = []
                for c in batch:
                    a_item = c.item_i if c.slot_a == "i" else c.item_j
                    b_item = c.item_j if c.slot_a == "i" else c.item_i
                    # render via the question, then prefill the assistant prefix
                    prompt = c.question.render(a_item, b_item)
                    texts.append(_prefill_text_for(self.tok, prompt, c.question.assistant_prefix))
                enc = self.tok(texts, return_tensors="pt", padding=True,
                               add_special_tokens=False).to(device)
                logits = _logits_from_output(self.model(**enc))[:, -1, :]
                ab = torch.stack([logits[:, a_id], logits[:, b_id]], dim=-1)
                p_a = torch.softmax(ab.float(), dim=-1)[:, 0].cpu().numpy()
                for c, pa in zip(batch, p_a):
                    pu = p_util_from_pick(float(pa), c.slot_a, c.question)
                    obs.append(EdgeObservation(
                        i=c.i, j=c.j, p_util=pu, mode="logit_local",
                        question_id=c.question.id, valence=c.question.valence,
                        slot_a=c.slot_a, phase=c.phase, round=c.round,
                        rank_distance=c.rank_distance, raw={"p": float(pa)}))
        return obs


def _prefill_text_for(tok, user_prompt, assistant_prefix):
    from .elicit import _apply_chat
    text = _apply_chat(tok, [{"role": "user", "content": user_prompt}], add_generation_prompt=True)
    return text + assistant_prefix
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: PASS (3 passed). (GPU paths are import-only here; not exercised on CI.)

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/oracle.py tests/test_oracle.py
git commit -m "feat: oracle — Comparison/EdgeObservation + slot/valence mapping + LocalLogitOracle"
```

---

## Task 10: oracle — OpenAIOracle (realtime logprob + sampling with n-param)

**Files:**
- Modify: `src/sentiment_utility/oracle.py`
- Test: `tests/test_oracle.py` (extend)

**Contract:** `OpenAIOracle(model, mode, n_samples, concurrency, calls_log, reasoning_effort)`
with `mode in {"logprob","sample"}`. Pure-function helpers `p_a_from_logprobs(top_logprobs, question)`
and `p_a_from_picks(picks)` are unit-tested on canned payloads; the async transport reuses
the retry/backoff from the old `elicit_mu_openai.py`. Sampling uses the chat-completions
`n` parameter (one request → N completions) with a per-model fallback to N separate calls.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle.py (append)
import math
from sentiment_utility.questions import Question
from sentiment_utility.oracle import p_a_from_logprobs, p_a_from_picks


def test_p_a_from_logprobs():
    q = Question(id="pos", template="x", valence=1, answers={"A": ["A"], "B": ["B"]})
    tops = [{"token": "A", "lp": math.log(0.75)}, {"token": "B", "lp": math.log(0.25)}]
    assert abs(p_a_from_logprobs(tops, q) - 0.75) < 1e-6


def test_p_a_from_picks_jeffreys():
    # 3 picks: A,A,B -> Jeffreys (2+0.5)/(3+1) = 0.625, also returns counts
    p, a, b = p_a_from_picks(["A", "A", "B"])
    assert abs(p - 0.625) < 1e-6 and a == 2 and b == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: FAIL (`p_a_from_logprobs` not defined)

- [ ] **Step 3: Add implementation to `oracle.py`**

```python
# src/sentiment_utility/oracle.py (append)
import math


def p_a_from_logprobs(top_logprobs, question) -> float:
    """top_logprobs: list of {token, lp}. Returns P(pick A) from the A/B labels'
    surface forms (first form per label)."""
    a_forms = {f.lower() for f in question.answers["A"]}
    b_forms = {f.lower() for f in question.answers["B"]}
    lpA = lpB = -math.inf
    for t in top_logprobs:
        tok = t["token"].strip().lower()
        if tok in a_forms:
            lpA = max(lpA, t["lp"])
        elif tok in b_forms:
            lpB = max(lpB, t["lp"])
    if lpA == -math.inf and lpB == -math.inf:
        return 0.5
    if lpA == -math.inf:
        return 0.0
    if lpB == -math.inf:
        return 1.0
    m = max(lpA, lpB)
    eA, eB = math.exp(lpA - m), math.exp(lpB - m)
    return eA / (eA + eB)


def p_a_from_picks(picks):
    """picks: list of 'A'/'B'/None. Returns (jeffreys_p_a, a_count, b_count)."""
    a = sum(1 for p in picks if p == "A")
    b = sum(1 for p in picks if p == "B")
    if a + b == 0:
        return 0.5, 0, 0
    return (a + 0.5) / (a + b + 1.0), a, b
```

Then add the async `OpenAIOracle` class. Port the retry/backoff and request code from the
current `scripts/elicit_mu_openai.py` (`_one_call`, `_one_call_sample`), adapting:
- render the prompt with `comparison.question.render(a_item, b_item)` (slot-aware);
- for sampling, issue **one** request with `n=n_samples` and read `resp.choices[k]`;
  on `BadRequestError` mentioning `n`, fall back to N separate calls;
- compute `p_pick_a` via `p_a_from_logprobs`/`p_a_from_picks`, then
  `p_util = p_util_from_pick(p_pick_a, slot_a, question)`;
- emit `EdgeObservation` with `mode="logprob"` or `"sample"`, `raw` carrying
  `{"lpA","lpB"}` or `{"wins_i","wins_j","n_samples","picks"}` where `wins_i`/`wins_j`
  are counts re-oriented to items (if `slot_a=="j"`, swap A/B counts; if valence==-1, swap
  again) so the fitter reads item-oriented wins directly.

```python
# src/sentiment_utility/oracle.py (append) — transport skeleton, full retry ported from old file
import asyncio


def _wins_to_items(a_count, b_count, slot_a, valence):
    """A/B pick counts -> (wins_i, wins_j) in utility orientation (item_i > item_j)."""
    # picks of item_i vs item_j
    wins_i_pick = a_count if slot_a == "i" else b_count
    wins_j_pick = b_count if slot_a == "i" else a_count
    # valence: picking an item under -1 means LOWER utility -> swap
    if valence == -1:
        wins_i_pick, wins_j_pick = wins_j_pick, wins_i_pick
    return wins_i_pick, wins_j_pick


class OpenAIOracle:
    def __init__(self, model, mode="logprob", n_samples=3, concurrency=40,
                 calls_log=None, reasoning_effort=None):
        from openai import AsyncOpenAI
        self.model = model
        self.mode = mode
        self.n_samples = n_samples
        self.calls_log = calls_log
        self.reasoning_effort = reasoning_effort
        self._client = AsyncOpenAI()
        self._sem = asyncio.Semaphore(concurrency)

    def compare(self, comparisons):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._compare_async(comparisons))
        finally:
            loop.close()

    async def _compare_async(self, comparisons):
        results = await asyncio.gather(*[self._one(c) for c in comparisons])
        return results

    async def _one(self, c):
        a_item = c.item_i if c.slot_a == "i" else c.item_j
        b_item = c.item_j if c.slot_a == "i" else c.item_i
        prompt = c.question.render(a_item, b_item)
        # ... retry/backoff body ported from elicit_mu_openai._one_call / _one_call_sample,
        #     issuing logprobs (max_completion_tokens=1, top_logprobs=20) or n-sampling ...
        if self.mode == "logprob":
            tops = await self._call_logprobs(prompt, c)   # returns [{token, lp}, ...]
            p_a = p_a_from_logprobs(tops, c.question)
            raw = {"lpA": _lp_of(tops, c.question, "A"), "lpB": _lp_of(tops, c.question, "B")}
            mode = "logprob"
        else:
            picks = await self._call_samples(prompt, c)   # returns list of 'A'/'B'/None
            p_a, a_cnt, b_cnt = p_a_from_picks([c.question.parse(p) if p else None for p in picks])
            wins_i, wins_j = _wins_to_items(a_cnt, b_cnt, c.slot_a, c.question.valence)
            raw = {"wins_i": wins_i, "wins_j": wins_j, "n_samples": self.n_samples}
            mode = "sample"
        p_util = p_util_from_pick(p_a, c.slot_a, c.question)
        return EdgeObservation(i=c.i, j=c.j, p_util=p_util, mode=mode,
                               question_id=c.question.id, valence=c.question.valence,
                               slot_a=c.slot_a, phase=c.phase, round=c.round,
                               rank_distance=c.rank_distance, raw=raw)
```

(Implementer: copy the concrete `_call_logprobs` / `_call_samples` retry bodies verbatim
from `scripts/elicit_mu_openai.py` `_one_call`/`_one_call_sample`, swapping the hard-coded
`PROMPT_TEMPLATE` for `prompt` and using `c.question` for parsing. Add `_lp_of(tops, q, label)`
returning the matched label's logprob or `None`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: PASS (5 passed). Network paths are not exercised on CI (only the pure helpers).

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/oracle.py tests/test_oracle.py
git commit -m "feat: oracle — OpenAIOracle (realtime logprob + n-param sampling), pure extractors tested"
```

---

## Task 11: oracle — Batch API execution path

**Files:**
- Modify: `src/sentiment_utility/oracle.py`
- Test: `tests/test_oracle.py` (extend)

**Contract:** `build_batch_requests(comparisons, model, mode, n_samples)` returns a list of
JSONL-ready dicts (`custom_id`, `method`, `url`, `body`) for the `/v1/chat/completions`
Batch endpoint, `custom_id = f"{i}_{j}_{slot_a}_{question_id}_{k}"`. `parse_batch_results`
maps returned `custom_id`→content back to `EdgeObservation`s. Submit/poll/download is a thin
wrapper (not unit-tested; exercised manually). This task tests the pure request builder and
result parser on canned data.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle.py (append)
from sentiment_utility.questions import Question
from sentiment_utility.oracle import Comparison, build_batch_requests


def test_build_batch_requests_logprob():
    q = Question(id="pos", template="A:{item_A} B:{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    comps = [Comparison(i=0, j=1, item_i="cat", item_j="dog", question=q, slot_a="i")]
    reqs = build_batch_requests(comps, model="gpt-4.1", mode="logprob", n_samples=1)
    assert reqs[0]["custom_id"] == "0_1_i_pos_0"
    assert reqs[0]["url"] == "/v1/chat/completions"
    assert reqs[0]["body"]["model"] == "gpt-4.1"
    assert reqs[0]["body"]["logprobs"] is True
    assert "cat" in reqs[0]["body"]["messages"][0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle.py::test_build_batch_requests_logprob -v`
Expected: FAIL (`build_batch_requests` not defined)

- [ ] **Step 3: Add implementation to `oracle.py`**

```python
# src/sentiment_utility/oracle.py (append)

def build_batch_requests(comparisons, model, mode, n_samples=1):
    reqs = []
    for c in comparisons:
        a_item = c.item_i if c.slot_a == "i" else c.item_j
        b_item = c.item_j if c.slot_a == "i" else c.item_i
        prompt = c.question.render(a_item, b_item)
        cid = f"{c.i}_{c.j}_{c.slot_a}_{c.question.id}_0"
        body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if mode == "logprob":
            body.update({"max_completion_tokens": 1, "logprobs": True, "top_logprobs": 20})
        else:
            body.update({"max_completion_tokens": 512, "n": n_samples})
        reqs.append({"custom_id": cid, "method": "POST",
                     "url": "/v1/chat/completions", "body": body})
    return reqs
```

Add `submit_batch(client, requests)`, `poll_batch(client, batch_id)`,
`parse_batch_results(raw_lines, comparisons_by_cid, mode)` returning `EdgeObservation`s
(reusing `p_a_from_logprobs`/`p_a_from_picks` + `_wins_to_items` + `p_util_from_pick`).
These wrap `client.files.create`, `client.batches.create(..., completion_window="24h")`,
`client.batches.retrieve`, `client.files.content`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_oracle.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/oracle.py tests/test_oracle.py
git commit -m "feat: oracle — Batch API request builder + result parser"
```

---

## Task 12: sampling — ELO active sampler

**Files:**
- Create: `src/sentiment_utility/sampling.py`
- Test: `tests/test_sampling.py`

**Contract:** `elo_active_sample(n, oracle, questions, R, m, floor, K, seed) -> list[EdgeObservation]`.
Each round builds `Comparison`s (random slot_a, random question from `questions`), calls
`oracle.compare`, ELO-updates ratings from `p_util`. Round 1 random partners; rounds 2..R
draw partner `j` for item `i` with prob `∝ p_ij(1-p_ij)` under current ratings plus uniform
floor `floor`. Tested with a deterministic fake oracle whose `p_util` is a logistic of a
hidden ground-truth score.

- [ ] **Step 1: Create the shared test scaffolding (fakes + conftest)**

```python
# tests/conftest.py — make the tests dir importable so tests can share helpers
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
```

```python
# tests/fakes.py
import numpy as np


class FakeOracle:
    """p_util(item_i > item_j) = logistic(score_i - score_j) from a hidden ground truth.
    Deterministic; used to drive sampling/integration tests without a model or network."""
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def compare(self, comparisons):
        from sentiment_utility.oracle import EdgeObservation
        obs = []
        for c in comparisons:
            d = self.scores[c.i] - self.scores[c.j]
            p = 1.0 / (1.0 + np.exp(-d))
            obs.append(EdgeObservation(i=c.i, j=c.j, p_util=float(p), mode="logprob",
                                       question_id=c.question.id, valence=c.question.valence,
                                       slot_a=c.slot_a, phase=c.phase, round=c.round))
        return obs
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_sampling.py
import numpy as np
from sentiment_utility.questions import Question
from sentiment_utility.sampling import elo_active_sample
from fakes import FakeOracle


def test_elo_sampler_covers_items_and_recovers_order():
    n = 20
    scores = np.linspace(-3, 3, n)
    rng = np.random.default_rng(0)
    scores = scores[rng.permutation(n)]
    q = [Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})]
    edges = elo_active_sample(n, FakeOracle(scores), q, R=5, m=6, floor=0.15, K=32, seed=0)
    seen = {e.i for e in edges} | {e.j for e in edges}
    assert seen == set(range(n))                 # every item compared
    # fit recovers the order
    from sentiment_utility.fit import fit_caseV_mle
    rows = [{"i": e.i, "j": e.j, "p_util": e.p_util, "mode": "logprob"} for e in edges]
    mu = fit_caseV_mle(rows, n=n, steps=1500, seed=0)["mu"]
    assert np.corrcoef(mu, scores)[0, 1] > 0.9
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: FAIL (`ModuleNotFoundError: sentiment_utility.sampling`)

- [ ] **Step 4: Write minimal implementation**

```python
# src/sentiment_utility/sampling.py
from __future__ import annotations

import numpy as np

from .oracle import Comparison


def _elo_expected(ri, rj, scale=400.0):
    return 1.0 / (1.0 + 10 ** ((rj - ri) / scale))


def _make_comparison(i, j, items, questions, rng, phase, rnd):
    q = questions[rng.integers(len(questions))]
    slot_a = "i" if rng.random() < 0.5 else "j"
    return Comparison(i=i, j=j, item_i=items[i] if items else str(i),
                      item_j=items[j] if items else str(j),
                      question=q, slot_a=slot_a, phase=phase, round=rnd)


def elo_active_sample(n, oracle, questions, R=5, m=5, floor=0.15, K=32, seed=0,
                      items=None):
    rng = np.random.default_rng(seed)
    ratings = np.zeros(n)
    all_obs = []
    seen = set()

    def submit(pairs, rnd):
        comps = [_make_comparison(i, j, items, questions, rng, "elo", rnd) for i, j in pairs]
        obs = oracle.compare(comps)
        for o in obs:
            all_obs.append(o)
            # ELO update from utility-oriented outcome
            exp_i = _elo_expected(ratings[o.i], ratings[o.j])
            ratings[o.i] += K * (o.p_util - exp_i)
            ratings[o.j] += K * ((1 - o.p_util) - (1 - exp_i))
        return obs

    for rnd in range(1, R + 1):
        pairs = []
        for i in range(n):
            if rnd == 1:
                partners = rng.choice([x for x in range(n) if x != i], size=min(m, n - 1),
                                      replace=False)
            else:
                # information weight ∝ p(1-p) under current ratings, + uniform floor
                d = (ratings[i] - ratings) / 400.0
                p = 1.0 / (1.0 + 10 ** (-d))
                info = p * (1 - p)
                info[i] = 0.0
                w = (1 - floor) * info + floor * (np.arange(n) != i)
                w = w / w.sum()
                partners = rng.choice(n, size=min(m, n - 1), replace=False, p=w)
            for jj in partners:
                key = (min(i, int(jj)), max(i, int(jj)))
                pairs.append((i, int(jj)))
                seen.add(key)
        submit(pairs, rnd)

    return all_obs
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/sentiment_utility/sampling.py tests/test_sampling.py tests/conftest.py tests/fakes.py
git commit -m "feat: sampling — batched ELO active sampler (info-weighted + uniform floor)"
```

---

## Task 13: sampling — reverse / triad / cross-question plans + edge writer

**Files:**
- Modify: `src/sentiment_utility/sampling.py`
- Test: `tests/test_sampling.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sampling.py (append)
from sentiment_utility.sampling import plan_reverse, plan_triads, plan_cross_question


def _qbank():
    from sentiment_utility.questions import Question
    return [Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]}),
            Question(id="neg", template="{item_A}{item_B}", valence=-1, answers={"A": ["A"], "B": ["B"]})]


def test_plan_reverse_flips_slot():
    obs_pairs = [(0, 1), (2, 3)]
    comps = plan_reverse(obs_pairs, items=["a", "b", "c", "d"], questions=_qbank(),
                         n_reverse=2, seed=0)
    assert len(comps) == 2
    for c in comps:
        assert c.phase == "reverse"


def test_plan_triads_three_edges_each():
    comps = plan_triads(order=list(range(10)), items=[str(x) for x in range(10)],
                        questions=_qbank(), n_triads=4, seed=0)
    assert len(comps) == 4 * 3        # three pairwise comparisons per triad
    assert all(c.phase == "triad" for c in comps)


def test_plan_cross_question_uses_nonprimary():
    comps = plan_cross_question(obs_pairs=[(0, 1)], items=["a", "b"], questions=_qbank(),
                                primary_id="pos", n_cross=1, seed=0)
    assert all(c.question.id != "pos" for c in comps)
    assert all(c.phase == "cross_question" for c in comps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: FAIL (`plan_reverse` not defined)

- [ ] **Step 3: Add implementation to `sampling.py`**

```python
# src/sentiment_utility/sampling.py (append)

def plan_reverse(obs_pairs, items, questions, n_reverse, seed=0):
    rng = np.random.default_rng(seed)
    pairs = list({(min(i, j), max(i, j)) for i, j in obs_pairs})
    if len(pairs) > n_reverse:
        idx = rng.choice(len(pairs), size=n_reverse, replace=False)
        pairs = [pairs[k] for k in idx]
    q = questions[0]   # primary question for position-bias measurement
    # query with slot_a = "j" (opposite of the canonical i-first)
    return [Comparison(i=i, j=j, item_i=items[i], item_j=items[j], question=q,
                       slot_a="j", phase="reverse") for i, j in pairs]


def plan_triads(order, items, questions, n_triads, seed=0):
    rng = np.random.default_rng(seed)
    n = len(order)
    q = questions[0]
    comps = []
    for _ in range(n_triads):
        # mix adjacent (cycles likeliest) and spread triples
        if rng.random() < 0.5 and n >= 3:
            r = rng.integers(0, n - 2)
            trip = [order[r], order[r + 1], order[r + 2]]
        else:
            trip = list(rng.choice(n, size=3, replace=False))
            trip = [order[t] for t in trip]
        a, b, c = trip
        for (x, y) in [(a, b), (b, c), (a, c)]:
            comps.append(Comparison(i=x, j=y, item_i=items[x], item_j=items[y],
                                    question=q, slot_a="i", phase="triad"))
    return comps


def plan_cross_question(obs_pairs, items, questions, primary_id, n_cross, seed=0):
    rng = np.random.default_rng(seed)
    others = [q for q in questions if q.id != primary_id]
    if not others:
        return []
    pairs = list({(min(i, j), max(i, j)) for i, j in obs_pairs})
    if len(pairs) > n_cross:
        idx = rng.choice(len(pairs), size=n_cross, replace=False)
        pairs = [pairs[k] for k in idx]
    comps = []
    for i, j in pairs:
        for q in others:
            comps.append(Comparison(i=i, j=j, item_i=items[i], item_j=items[j],
                                    question=q, slot_a="i", phase="cross_question"))
    return comps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/sampling.py tests/test_sampling.py
git commit -m "feat: sampling — reverse/triad/cross-question phase plans"
```

---

## Task 14: run_elicitation — single entry point + end-to-end integration test

**Files:**
- Create: `scripts/run_elicitation.py`
- Test: `tests/test_run_elicitation.py`

**Contract:** `run_elicitation(oracle, items, questions, out_dir, elo_cfg, phase_cfg, seed)`
runs ELO → reverse → triad → cross, writes `edges.jsonl` (via `JsonlAppender`,
`EdgeObservation.to_record(items)`), fits Case V MLE on `elo` edges, computes the panel,
and writes `mu.json`, `panel.json`, `metrics.json`. The CLI builds the backend oracle
(`--backend local|openai`, `--api-exec`, `--mode`, `--question-bank`). The integration
test drives the whole thing with the `FakeOracle` from Task 12 and asserts the outputs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_elicitation.py
import json
import numpy as np
from sentiment_utility.questions import Question


def test_end_to_end_with_fake_oracle(tmp_path):
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location("run_elicitation", "scripts/run_elicitation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from fakes import FakeOracle
    n = 15
    rng = np.random.default_rng(1)
    scores = rng.normal(size=n)
    items = [f"item{k}" for k in range(n)]
    questions = [Question(id="pos", template="{item_A}{item_B}", valence=1,
                          answers={"A": ["A"], "B": ["B"]})]
    out = tmp_path / "run"
    mod.run_elicitation(FakeOracle(scores), items, questions, out,
                        elo_cfg=dict(R=5, m=6, floor=0.15, K=32),
                        phase_cfg=dict(n_reverse=0, n_triads=10, n_cross=0),
                        seed=0)
    assert (out / "edges.jsonl").exists()
    panel = json.loads((out / "panel.json").read_text())
    assert 0.0 <= panel["decisiveness"]["point"] <= 1.0
    mu = json.loads((out / "mu.json").read_text())
    assert len(mu) == n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_elicitation.py -v`
Expected: FAIL (file not found)

- [ ] **Step 3: Write the implementation**

```python
# scripts/run_elicitation.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentiment_utility.io_utils import JsonlAppender, git_commit, jsonable, load_items, setup_logging
from sentiment_utility.questions import load_question_bank
from sentiment_utility.sampling import (
    elo_active_sample, plan_reverse, plan_triads, plan_cross_question,
)
from sentiment_utility.fit import fit_caseV_mle
from sentiment_utility.panel import compute_panel


def _obs_to_row(o, items):
    return o.to_record(items)


def run_elicitation(oracle, items, questions, out_dir, elo_cfg, phase_cfg, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_log = JsonlAppender(out_dir / "edges.jsonl")
    n = len(items)

    elo_obs = elo_active_sample(n, oracle, questions, items=items, seed=seed, **elo_cfg)
    for o in elo_obs:
        edges_log.write(_obs_to_row(o, items))

    rows_elo = [{"i": o.i, "j": o.j, "p_util": o.p_util, "mode": o.mode, **o.raw}
                for o in elo_obs]
    mu = fit_caseV_mle(rows_elo, n=n, seed=seed)["mu"]
    order = list(np.argsort(-mu))
    obs_pairs = [(o.i, o.j) for o in elo_obs]

    # non-adaptive sweep
    extra = []
    if phase_cfg.get("n_reverse"):
        extra += oracle.compare(plan_reverse(obs_pairs, items, questions,
                                             phase_cfg["n_reverse"], seed))
    if phase_cfg.get("n_triads"):
        extra += oracle.compare(plan_triads(order, items, questions,
                                            phase_cfg["n_triads"], seed))
    if phase_cfg.get("n_cross"):
        extra += oracle.compare(plan_cross_question(obs_pairs, items, questions,
                                                    questions[0].id, phase_cfg["n_cross"], seed))
    for o in extra:
        edges_log.write(_obs_to_row(o, items))
    edges_log.close()

    edges_by_phase = _bucket_for_panel(elo_obs, extra)
    panel = compute_panel(edges_by_phase, n=n, seed=seed)

    (out_dir / "mu.json").write_text(json.dumps(
        {it: float(v) for it, v in zip(items, mu)}, indent=2))
    (out_dir / "panel.json").write_text(json.dumps(jsonable(panel), indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(jsonable({
        "commit": git_commit(), "n_items": n,
        "n_elo": len(elo_obs), "n_extra": len(extra),
    }), indent=2))
    return panel


def _bucket_for_panel(elo_obs, extra):
    elo = [{"i": o.i, "j": o.j, "p_util": o.p_util, "mode": o.mode, **o.raw} for o in elo_obs]
    fwd = {(o.i, o.j): o.p_util for o in elo_obs}
    reverse, cross = [], []
    triad_putil = []      # p_util in strict emission order: [(a,b),(b,c),(a,c), ...]
    for o in extra:
        if o.phase == "reverse" and (o.i, o.j) in fwd:
            # p_fwd = P(pick i | i first) = fwd p_util; p_rev = P(pick i | i second)
            reverse.append({"p_fwd": fwd[(o.i, o.j)], "p_rev": o.p_util})
        elif o.phase == "triad":
            triad_putil.append(o.p_util)
        elif o.phase == "cross_question" and (o.i, o.j) in fwd:
            cross.append({"p_util_a": fwd[(o.i, o.j)], "p_util_b": o.p_util})
    return {"elo": elo, "reverse": reverse,
            "triad": _assemble_triads(triad_putil), "cross": cross}


def _assemble_triads(triad_putil):
    """Chunk the ordered triad p_util list by 3: emitted as (a,b),(b,c),(a,c) per triad.
    Convert the (a,c) edge to the (c,a) direction for the cycle-mass formula."""
    out = []
    for t in range(0, len(triad_putil) - 2, 3):
        p_ab, p_bc, p_ac = triad_putil[t], triad_putil[t + 1], triad_putil[t + 2]
        out.append((p_ab, p_bc, 1.0 - p_ac))
    return out


def _build_oracle(args, items, questions, out_dir):
    if args.backend == "local":
        from sentiment_utility.elicit import load_model
        from sentiment_utility.oracle import LocalLogitOracle
        tok, model = load_model(args.model_id, revision=args.revision,
                                load_in_4bit=args.load_in_4bit)
        return LocalLogitOracle(tok, model, batch_size=args.batch_size)
    from sentiment_utility.oracle import OpenAIOracle
    calls_log = JsonlAppender(out_dir / "calls.jsonl")
    return OpenAIOracle(args.model_id, mode=args.mode, n_samples=args.samples,
                        concurrency=args.concurrency, calls_log=calls_log,
                        reasoning_effort=args.reasoning_effort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "openai"], required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--items-path", default="config/items_500.yaml")
    ap.add_argument("--question-bank", default="config/questions_default.jsonl")
    ap.add_argument("--out-root", default="runs/elicit")
    ap.add_argument("--mode", choices=["logprob", "sample"], default="logprob")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--api-exec", choices=["realtime", "batch", "auto"], default="auto")
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--n-reverse", type=int, default=500)
    ap.add_argument("--n-triads", type=int, default=1000)
    ap.add_argument("--n-cross", type=int, default=500)
    args = ap.parse_args()

    items = load_items(args.items_path)
    questions = load_question_bank(args.question_bank)
    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    oracle = _build_oracle(args, items, questions, out_dir)
    panel = run_elicitation(
        oracle, items, questions, out_dir,
        elo_cfg=dict(R=args.R, m=args.m, floor=0.15, K=32),
        phase_cfg=dict(n_reverse=args.n_reverse, n_triads=args.n_triads, n_cross=args.n_cross),
        seed=0,
    )
    print(json.dumps(jsonable(panel), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_elicitation.py -v`
Expected: PASS. If `_assemble_triads` ordering is brittle, the integration test uses
`n_triads=10` with the FakeOracle — confirm tuples are well-formed (length-3) and panel
`transitivity_triad.point` is finite.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_elicitation.py tests/test_run_elicitation.py
git commit -m "feat: run_elicitation — unified entry point + end-to-end fake-oracle test"
```

---

## Task 15: build_coherence — panel-driven master CSV

**Files:**
- Create: `scripts/build_coherence.py`
- Test: `tests/test_build_coherence.py`

**Contract:** `panel_row_from_edges(edges_path, items_path, B)` loads an `edges.jsonl`,
buckets by phase, fits + computes the panel, returns a flat dict (each metric →
`point`, `meas_lo`, `meas_hi`, `gen_lo`, `gen_hi`). `build(run_specs)` iterates a registry
(ported from `build_coherence_v4.build`) and writes `results/coherence_all_v5.csv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_coherence.py
import json
import math
import numpy as np
import importlib.util
from pathlib import Path


def _load_mod():
    spec = importlib.util.spec_from_file_location("build_coherence", "scripts/build_coherence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_panel_row_from_edges(tmp_path):
    mod = _load_mod()
    # write a tiny edges.jsonl + items.yaml
    n = 6
    scores = np.linspace(-2, 2, n)
    items_path = tmp_path / "items.yaml"
    items_path.write_text("items:\n" + "".join(f"  - item{k}\n" for k in range(n)))
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = float(0.5 * (1 + math.erf((scores[i] - scores[j]) / 2.0)))
            rows.append({"i": i, "j": j, "p_util": p, "mode": "logprob", "phase": "elo"})
    edges_path = tmp_path / "edges.jsonl"
    edges_path.write_text("\n".join(json.dumps(r) for r in rows))
    row = mod.panel_row_from_edges(edges_path, items_path, B=40)
    assert 0.0 <= row["decisiveness_point"] <= 1.0
    assert row["decisiveness_meas_lo"] <= row["decisiveness_point"] <= row["decisiveness_meas_hi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_build_coherence.py -v`
Expected: FAIL (file not found)

- [ ] **Step 3: Write the implementation**

```python
# scripts/build_coherence.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentiment_utility.io_utils import load_items
from sentiment_utility.panel import compute_panel


def _load_edges(edges_path):
    rows = []
    for line in Path(edges_path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _bucket(rows):
    """rows are in file (emission) order; triad edges come in (a,b),(b,c),(a,c) groups."""
    elo = [r for r in rows if r.get("phase", "elo") == "elo"]
    fwd = {(r["i"], r["j"]): r["p_util"] for r in elo}
    reverse, cross, triad_putil = [], [], []
    for r in rows:
        ph = r.get("phase")
        if ph == "reverse" and (r["i"], r["j"]) in fwd:
            reverse.append({"p_fwd": fwd[(r["i"], r["j"])], "p_rev": r["p_util"]})
        elif ph == "triad":
            triad_putil.append(r["p_util"])
        elif ph == "cross_question" and (r["i"], r["j"]) in fwd:
            cross.append({"p_util_a": fwd[(r["i"], r["j"])], "p_util_b": r["p_util"]})
    triads = [(triad_putil[t], triad_putil[t + 1], 1.0 - triad_putil[t + 2])
              for t in range(0, len(triad_putil) - 2, 3)]
    return {"elo": elo, "reverse": reverse, "triad": triads, "cross": cross}


def _flatten(panel):
    out = {}
    for key, v in panel.items():
        out[f"{key}_point"] = v["point"]
        out[f"{key}_meas_lo"], out[f"{key}_meas_hi"] = v["meas_ci"]
        out[f"{key}_gen_lo"], out[f"{key}_gen_hi"] = v["gen_ci"]
    return out


def panel_row_from_edges(edges_path, items_path, B=200):
    items = load_items(items_path)
    rows = _load_edges(edges_path)
    panel = compute_panel(_bucket(rows), n=len(items), B=B)
    return _flatten(panel)


def main():
    # registry of (group, model, family, role, edges_path, items_path);
    # port the run list from build_coherence_v4.build(), pointing at each run's edges.jsonl.
    raise SystemExit("populate the run registry then write results/coherence_all_v5.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_build_coherence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/build_coherence.py tests/test_build_coherence.py
git commit -m "feat: build_coherence — panel-driven master CSV from edges.jsonl (v5)"
```

---

## Task 16: cleanup — relocate helpers, shim old scripts, retire dead fit code

**Files:**
- Modify: `scripts/run_character.py`, `scripts/elicit_mu.py`, `scripts/elicit_mu_openai.py`,
  `scripts/fit_bayesian.py`, `src/sentiment_utility/efficient.py`

- [ ] **Step 1: Point `run_character.py` at the relocated helpers**

Replace the local `_git_commit`, `_jsonable`, `_load_items`, `_setup_logging` definitions
in `scripts/run_character.py` (lines 28-64) with imports, keeping the old names as aliases
so existing call sites and `from run_character import …` keep working during transition:

```python
from sentiment_utility.io_utils import (
    git_commit as _git_commit, jsonable as _jsonable,
    load_items as _load_items, setup_logging as _setup_logging,
)
```

- [ ] **Step 2: Run the existing suite to confirm nothing broke**

Run: `uv run pytest -q`
Expected: PASS (all prior tests green; helper relocation is behavior-preserving)

- [ ] **Step 3: Shim the old elicit scripts**

Replace the bodies of `scripts/elicit_mu.py` and `scripts/elicit_mu_openai.py` with a
deprecation pointer (keep the files so docs/links don't 404):

```python
# scripts/elicit_mu.py
import sys
print("DEPRECATED: use scripts/run_elicitation.py --backend local "
      "(see docs/superpowers/specs/2026-05-29-elicitation-redesign-design.md)",
      file=sys.stderr)
sys.exit(2)
```

```python
# scripts/elicit_mu_openai.py
import sys
print("DEPRECATED: use scripts/run_elicitation.py --backend openai", file=sys.stderr)
sys.exit(2)
```

- [ ] **Step 4: Retire subsumed fit code**

In `src/sentiment_utility/efficient.py`, delete `spacing_pass` (subsumed by the ELO
uniform-floor anchors). Keep `rank_by_quicksort` and `fit_thurstone_sparse` ONLY if still
imported by `run_character.py`/`run_scale.py`; otherwise delete them too. Grep first:

Run: `grep -rn "spacing_pass\|fit_thurstone_sparse\|from scripts.fit_bayesian\|import fit_bayesian" scripts src tests`

Delete `scripts/fit_bayesian.py` (MAP/HMC/τ/Jeffreys all superseded by `fit.py`). Update
any remaining importers to `sentiment_utility.fit`.

- [ ] **Step 5: Run the suite + commit**

```bash
uv run pytest -q
git add -A
git commit -m "chore: relocate helpers, shim deprecated elicit scripts, retire MAP/HMC + spacing_pass"
```

---

## Task 17: dense sanity harness on the question-bank interface

**Files:**
- Modify: `src/sentiment_utility/run.py` (or `scripts/`), test: `tests/test_dense_sanity.py`

**Contract:** the 25-item dense pipeline elicits BOTH orderings of every pair through the
unified `LocalLogitOracle` + question bank, builds the full preference matrix, and reports
the panel — validating the A/B instrument itself. Reuses `rank_by_quicksort`? No: dense =
all-pairs, no sort needed. Provide `dense_compare_all(oracle, items, questions)` returning
edges for every ordered pair, both slots, then reuse `compute_panel`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dense_sanity.py
import numpy as np
from sentiment_utility.questions import Question
from sentiment_utility.run import dense_compare_all
from fakes import FakeOracle


def test_dense_compare_all_covers_all_ordered_pairs():
    n = 5
    items = [f"i{k}" for k in range(n)]
    q = [Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})]
    edges = dense_compare_all(FakeOracle(np.arange(n)), items, q)
    pairs = {(e.i, e.j) for e in edges}
    assert len(pairs) == n * (n - 1)     # every ordered pair present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dense_sanity.py -v`
Expected: FAIL (`dense_compare_all` not defined)

- [ ] **Step 3: Implement `dense_compare_all` in `run.py`**

```python
# src/sentiment_utility/run.py (add near top, after imports)
from .oracle import Comparison


def dense_compare_all(oracle, items, questions):
    n = len(items)
    q = questions[0]
    comps = [Comparison(i=i, j=j, item_i=items[i], item_j=items[j], question=q,
                        slot_a="i", phase="elo")
             for i in range(n) for j in range(n) if i != j]
    return oracle.compare(comps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dense_sanity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sentiment_utility/run.py tests/test_dense_sanity.py
git commit -m "feat: dense sanity harness on unified oracle/question-bank interface"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Smoke-test the entry point on a tiny local model (GPU pod)**

Run: `uv run python scripts/run_elicitation.py --backend local --model-id google/gemma-3-1b-it --name smoke --items-path config/items.yaml --n-reverse 50 --n-triads 50 --n-cross 50`
Expected: writes `runs/elicit/smoke/{edges.jsonl,mu.json,panel.json,metrics.json}`; panel
decisiveness in [0.5,1] after `0.5+0.5·D`, transitivity near 1 for a coherent model.

- [ ] **Update README** with the new unified workflow (one paragraph + the command above),
  and link the spec. Commit.

---

## Notes for the implementer

- **GPU vs CPU:** `fit.py` auto-selects CUDA; tests run on CPU. The bootstrap is the only
  heavy step — keep `B` modest in tests (≤200).
- **Weekend re-run:** after merge, re-elicit each model in the `build_coherence` registry
  with `run_elicitation` so every run has a phase-tagged `edges.jsonl`; then run
  `build_coherence.py` to emit `coherence_all_v5.csv`. Existing `edges.jsonl` (gpt-5.x,
  local) can be panel-scored immediately for decisiveness/transitivity_fas/unidim_fit, but
  reverse/triad/cross metrics will be blank until re-run.
- **Logging discipline (CLAUDE.md):** `run_elicitation` already writes `edges.jsonl` +
  `calls.jsonl` + `metrics.json` with the git commit; keep that. Upload run logs to the
  `arcadia-impact` HF org per project convention.
