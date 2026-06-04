# HF Dataset Hosting + `question-consistency` Rebrand — Implementation Plan

> **For agentic workers:** built for execution with **superpowers:codex-driven-development** (Codex builds, Claude reviews). Steps use checkbox (`- [ ]`) syntax. Tasks tagged **[CODE]** are Codex build tasks; **[OPS]** are orchestrator-run (network/GitHub actions Codex should not do).

**Goal:** Host the large datasets as parquet HF Datasets with on-demand + arbitrary-external loading, and rebrand the project `sentiment-utility` → `question-consistency`.

**Architecture:** A new `question_consistency/datasets.py` holds `resolve_items(ref)` with layered resolution (local YAML → known name → `hf://file` → `hf-dataset:` external); `io_utils.load_items` delegates to it so the whole pipeline gets HF resolution for free. An uploader publishes the three large YAMLs as parquet configs. The package directory and all references are renamed first, mechanically, and proven green by `pytest`.

**Tech Stack:** Python 3.11+, `uv`, `datasets`, `huggingface_hub`, `pyyaml`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-06-04-hf-dataset-hosting-and-rebrand-design.md`

---

## File Structure

- `src/question_consistency/` — renamed from `src/sentiment_utility/` (Task 1).
- `src/question_consistency/datasets.py` — **new**: `HF_DATASET_REPO`, `REGISTRY`, `resolve_items`, `read_yaml_items` (Tasks 2–3).
- `src/question_consistency/io_utils.py` — `load_items` delegates to `resolve_items` (Task 4).
- `scripts/upload_datasets_hf.py` — **new**: YAML → parquet config `push_to_hub` (Task 5).
- `tests/test_datasets_resolver.py` — **new**: resolver unit tests (Tasks 2–3).
- `tests/test_upload_datasets.py` — **new**: YAML→Dataset conversion unit test (Task 5).
- `config/datasets/{items_500,items_2000,curated_concepts}.yaml` — **deleted** from git after upload (Task 7).
- `README.md`, `pyproject.toml` — rebrand + Datasets section (Tasks 1, 8).

---

## Task 1: Rebrand `sentiment_utility` → `question_consistency` [CODE]

**Files:** `git mv src/sentiment_utility src/question_consistency`; rewrite refs across `src/`, `scripts/`, `tests/`; `pyproject.toml`; `README.md`.

- [ ] **Step 1: Move the package directory**

```bash
git mv src/sentiment_utility src/question_consistency
```

- [ ] **Step 2: Rewrite all identifier references (imports, strings)**

```bash
# every .py under src/scripts/tests that mentions the old package
grep -rl "sentiment_utility" src scripts tests | xargs sed -i 's/sentiment_utility/question_consistency/g'
# verify none remain
grep -rn "sentiment_utility" src scripts tests || echo "OK: no sentiment_utility references left"
```

- [ ] **Step 3: Update pyproject + README title**

In `pyproject.toml`: set `name = "question-consistency"` and `[tool.hatch.build.targets.wheel] packages = ["src/question_consistency"]` (leave `pythonpath = ["src"]`).
In `README.md`: change the first line `# Sentiment Utility` → `# Question Consistency`, and any prose that names the project "sentiment-utility" → "question-consistency" (do NOT edit dated files under `docs/superpowers/`).

```bash
sed -i 's/^name = "sentiment-utility"/name = "question-consistency"/' pyproject.toml
sed -i 's#packages = \["src/sentiment_utility"\]#packages = ["src/question_consistency"]#' pyproject.toml
sed -i '1s/# Sentiment Utility/# Question Consistency/' README.md
```

- [ ] **Step 4: Rebuild env + run the full test suite**

Run: `uv sync && uv run pytest -q`
Expected: package reinstalls as `question-consistency`; **all tests pass**. If any test still imports `sentiment_utility`, fix it (Step 2 should have caught it) and re-run.

- [ ] **Step 5: Import smoke**

Run: `uv run python -c "import question_consistency; from question_consistency.io_utils import load_items; print('rebrand OK')"`
Expected: `rebrand OK`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rebrand sentiment_utility -> question_consistency (package + project name)"
```

---

## Task 2: Dataset resolver — local YAML + registry parsing (TDD) [CODE]

**Files:** Create `src/question_consistency/datasets.py`; Test `tests/test_datasets_resolver.py`.

- [ ] **Step 1: Write failing tests for local + registry resolution**

```python
# tests/test_datasets_resolver.py
import pytest
from question_consistency import datasets as D

def test_local_items_yaml(tmp_path):
    p = tmp_path / "x.yaml"; p.write_text("items:\n- a\n- b\n")
    assert D.resolve_items(str(p)) == ["a", "b"]

def test_local_concepts_yaml(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("concepts:\n- x\n- y\n")
    assert D.resolve_items(str(p)) == ["x", "y"]

def test_bare_known_name_pulls_hf(monkeypatch):
    calls = {}
    monkeypatch.setattr(D, "_from_hf_config", lambda name: calls.setdefault("name", name) or ["i1"])
    assert D.resolve_items("items_2000") == ["i1"]
    assert calls["name"] == "items_2000"

def test_missing_local_path_with_registered_basename(monkeypatch):
    monkeypatch.setattr(D, "_from_hf_config", lambda name: [f"hf:{name}"])
    assert D.resolve_items("config/datasets/items_2000.yaml") == ["hf:items_2000"]

def test_unresolvable_raises():
    with pytest.raises(FileNotFoundError):
        D.resolve_items("config/datasets/does_not_exist.yaml")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_datasets_resolver.py -q`
Expected: FAIL (module `question_consistency.datasets` not found).

- [ ] **Step 3: Implement the module (local + registry; HF helpers stubbed for Task 3)**

```python
# src/question_consistency/datasets.py
from __future__ import annotations
from pathlib import Path
import yaml

HF_DATASET_REPO = "arcadia-impact/question-consistency-datasets"
REGISTRY = {"items_500", "items_2000", "curated_concepts"}  # hosted on HF as parquet configs


def read_yaml_items(path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("items", "concepts"):
        if key in data:
            return list(data[key])
    raise ValueError(f"{path}: YAML has neither 'items' nor 'concepts' key")


def _from_hf_config(name: str) -> list[str]:
    from datasets import load_dataset
    return list(load_dataset(HF_DATASET_REPO, name=name, split="train")["item"])


def _from_hf_dataset(spec: str) -> list[str]:
    # hf-dataset:<repo>:<split>:<column>
    rest = spec[len("hf-dataset:"):]
    parts = rest.split(":")
    if len(parts) != 3:
        raise ValueError("hf-dataset ref must be hf-dataset:<repo>:<split>:<column>")
    repo, split, column = parts
    from datasets import load_dataset
    return list(load_dataset(repo, split=split)[column])


def _from_hf_file(spec: str) -> list[str]:
    # hf://<owner>/<repo>/<path-in-repo>
    from huggingface_hub import hf_hub_download
    segs = spec[len("hf://"):].split("/")
    if len(segs) < 3:
        raise ValueError("hf:// ref must be hf://<owner>/<repo>/<path>")
    repo_id = "/".join(segs[:2])
    local = hf_hub_download(repo_id=repo_id, filename="/".join(segs[2:]), repo_type="dataset")
    return read_yaml_items(local)


def resolve_items(ref) -> list[str]:
    """Resolve a dataset reference to a list of item strings.

    Forms: local YAML path | bare known name | hf://<owner>/<repo>/<file> |
    hf-dataset:<repo>:<split>:<column>. A *missing* local path whose basename is a
    registered name (e.g. config/datasets/items_2000.yaml) auto-pulls from HF."""
    ref = str(ref)
    if ref.startswith("hf-dataset:"):
        return _from_hf_dataset(ref)
    if ref.startswith("hf://"):
        return _from_hf_file(ref)
    p = Path(ref)
    if p.exists():
        return read_yaml_items(p)
    name = p.name[:-5] if p.name.endswith(".yaml") else p.name
    if ref in REGISTRY:
        return _from_hf_config(ref)
    if name in REGISTRY:
        return _from_hf_config(name)
    raise FileNotFoundError(
        f"cannot resolve dataset ref {ref!r}: not a local file, not a known name "
        f"{sorted(REGISTRY)}, and not an hf://… or hf-dataset:… reference")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_datasets_resolver.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/question_consistency/datasets.py tests/test_datasets_resolver.py
git commit -m "feat(datasets): resolve_items with local YAML + HF-config registry"
```

---

## Task 3: External HF dataset + hf:// file resolution (TDD) [CODE]

**Files:** Modify `tests/test_datasets_resolver.py`. (`datasets.py` helpers already written in Task 2; here we test them with monkeypatched backends so CI needs no network.)

- [ ] **Step 1: Add failing tests for `hf-dataset:` and `hf://`**

```python
# append to tests/test_datasets_resolver.py
def test_hf_dataset_external(monkeypatch):
    seen = {}
    class _DS(dict):
        pass
    def fake_load_dataset(repo, split=None, name=None):
        seen.update(repo=repo, split=split, name=name)
        return {"title": ["t1", "t2"]}
    import datasets as real_datasets
    monkeypatch.setattr(real_datasets, "load_dataset", fake_load_dataset)
    assert D.resolve_items("hf-dataset:Shengtao/recipe:train:title") == ["t1", "t2"]
    assert seen == {"repo": "Shengtao/recipe", "split": "train", "name": None}

def test_hf_file(monkeypatch, tmp_path):
    f = tmp_path / "remote.yaml"; f.write_text("items:\n- q\n- r\n")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename, repo_type: str(f))
    assert D.resolve_items("hf://arcadia-impact/question-consistency-datasets/items.yaml") == ["q", "r"]
```

- [ ] **Step 2: Run to verify pass (helpers exist from Task 2)**

Run: `uv run pytest tests/test_datasets_resolver.py -q`
Expected: 7 passed. (If `_from_hf_dataset` imports `load_dataset` as a local name that monkeypatching `datasets.load_dataset` doesn't intercept, change the helper to `import datasets; datasets.load_dataset(...)` so the patch applies, and re-run.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_datasets_resolver.py src/question_consistency/datasets.py
git commit -m "test(datasets): cover hf-dataset: and hf:// resolution"
```

---

## Task 4: Wire `load_items` + consolidate duplicate loaders [CODE]

**Files:** Modify `src/question_consistency/io_utils.py`; `src/question_consistency/data.py`; `scripts/{run_character,run_audit,refit_edges,validate_method}.py`; `scripts/build_dataset.py`.

- [ ] **Step 1: Delegate `io_utils.load_items` to the resolver**

Replace the body of `load_items` in `src/question_consistency/io_utils.py`:

```python
def load_items(path) -> list[str]:
    from .datasets import resolve_items
    return resolve_items(path)
```

(Leave the `import yaml` in io_utils if other functions use it; otherwise remove.)

- [ ] **Step 2: Point `data.py.load_items` and script-local `_load_items` at the resolver**

In `src/question_consistency/data.py`, make `load_items` delegate identically:

```python
def load_items(path: str) -> list[str]:
    from .datasets import resolve_items
    return resolve_items(path)
```

In `scripts/run_character.py`, `scripts/run_audit.py`, `scripts/refit_edges.py`, `scripts/validate_method.py`: replace each local `_load_items` definition with an import and alias near the top:

```python
from question_consistency.io_utils import load_items as _load_items
```

(Delete the old `def _load_items(...)` body in each.)

- [ ] **Step 3: Route `build_dataset.load_curated` through the resolver**

In `scripts/build_dataset.py`, `load_curated` currently reads the YAML directly. Change it to obtain the concept strings via the resolver so it works whether `curated_concepts.yaml` is local or on HF:

```python
def load_curated(path) -> list[tuple[str, None]]:
    from question_consistency.io_utils import load_items
    return [(item, None) for item in load_items(path)]
```

(Keep the call site `load_curated(Path("config/datasets/curated_concepts.yaml"))` — a missing local file now auto-resolves to the HF config.)

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (resolver tests + existing tests; existing tests use the tiny in-repo `items.yaml`, which still resolves locally).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: route all item loading through resolve_items"
```

---

## Task 5: Uploader script `scripts/upload_datasets_hf.py` (TDD for the converter) [CODE]

**Files:** Create `scripts/upload_datasets_hf.py`; Test `tests/test_upload_datasets.py`.

- [ ] **Step 1: Failing test for the YAML→Dataset converter**

```python
# tests/test_upload_datasets.py
import importlib.util
from pathlib import Path

def _load_mod():
    p = Path("scripts/upload_datasets_hf.py")
    spec = importlib.util.spec_from_file_location("upload_datasets_hf", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def test_yaml_to_dataset_items(tmp_path):
    m = _load_mod()
    p = tmp_path / "items_500.yaml"; p.write_text("items:\n- a\n- b\n- c\n")
    ds = m.yaml_to_dataset(p)
    assert ds.column_names == ["item"]
    assert ds["item"] == ["a", "b", "c"]

def test_yaml_to_dataset_concepts(tmp_path):
    m = _load_mod()
    p = tmp_path / "curated_concepts.yaml"; p.write_text("concepts:\n- x\n- y\n")
    ds = m.yaml_to_dataset(p)
    assert ds["item"] == ["x", "y"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_upload_datasets.py -q`
Expected: FAIL (file/function missing).

- [ ] **Step 3: Implement the uploader**

```python
# scripts/upload_datasets_hf.py
"""Publish the large datasets to a HuggingFace dataset repo as parquet configs.

Each YAML (items: or concepts:) becomes a single-column `item` Dataset pushed as its
own config. xet upload is disabled in-process (the xet backend stalled during the
small-N run; classic LFS works).

  uv run python scripts/upload_datasets_hf.py            # default: all registered, to arcadia repo
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # MUST precede hf imports

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from question_consistency.datasets import HF_DATASET_REPO, REGISTRY, read_yaml_items  # noqa: E402


def yaml_to_dataset(path):
    from datasets import Dataset
    return Dataset.from_dict({"item": read_yaml_items(path)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=HF_DATASET_REPO)
    ap.add_argument("--datasets", default=",".join(sorted(REGISTRY)),
                    help="comma-separated config names (must exist as config/datasets/<name>.yaml)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    HfApi().create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        path = REPO / "config" / "datasets" / f"{name}.yaml"
        if not path.exists():
            print(f"SKIP {name}: {path} not found"); continue
        ds = yaml_to_dataset(path)
        ds.push_to_hub(args.repo, config_name=name, split="train")
        print(f"PUSHED {name}: {len(ds)} rows -> {args.repo} (config={name})")
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run converter test to verify pass**

Run: `uv run pytest tests/test_upload_datasets.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/upload_datasets_hf.py tests/test_upload_datasets.py
git commit -m "feat: upload_datasets_hf.py (YAML -> parquet HF config, xet disabled)"
```

---

## Task 6: Publish datasets to HF [OPS — orchestrator runs; needs network + arcadia write]

- [ ] **Step 1: Run the uploader (files still present in git at this point)**

Run: `uv run python scripts/upload_datasets_hf.py --datasets items_500,items_2000,curated_concepts`
Expected: three `PUSHED …` lines; repo `arcadia-impact/question-consistency-datasets` created public.

- [ ] **Step 2: Verify configs are pullable**

Run:
```bash
uv run python -c "
from datasets import load_dataset
for n in ['items_500','items_2000','curated_concepts']:
    d=load_dataset('arcadia-impact/question-consistency-datasets', name=n, split='train')
    print(n, len(d), d['item'][:2])"
```
Expected: 500 / 2000 / 268 rows with sample items.

- [ ] **Step 3: Write the dataset card** (`README.md` in the HF repo) documenting the configs, `{item}` schema, `load_dataset` usage, and provenance. Upload via `HfApi().upload_file(..., repo_type="dataset")` with `HF_HUB_DISABLE_XET=1`.

---

## Task 7: Remove large YAMLs from git + verify HF fallback [CODE+OPS]

**Files:** delete `config/datasets/{items_500,items_2000,curated_concepts}.yaml`.

- [ ] **Step 1: Confirm they exist on HF (Task 6 done) then git-rm**

```bash
git rm config/datasets/items_500.yaml config/datasets/items_2000.yaml config/datasets/curated_concepts.yaml
```

- [ ] **Step 2: Verify back-compat resolution from HF (file now absent)**

Run:
```bash
uv run python -c "
from question_consistency.io_utils import load_items
print('by path :', len(load_items('config/datasets/items_2000.yaml')))
print('by name :', len(load_items('items_500')))"
```
Expected: `by path : 2000` and `by name : 500` (both pulled from HF).

- [ ] **Step 3: Verify tiny in-repo sets still load locally**

Run: `uv run python -c "from question_consistency.io_utils import load_items; print(len(load_items('config/datasets/items.yaml')), len(load_items('config/datasets/leetcode_problems.yaml')))"`
Expected: `25 40`.

- [ ] **Step 4: Run full suite + commit**

Run: `uv run pytest -q` (expected: green)
```bash
git add -A
git commit -m "chore: move large datasets to HF (removed from git; auto-pulled on demand)"
```

---

## Task 8: README Datasets section + GitHub repo rename [CODE+OPS]

- [ ] **Step 1: Add a "Datasets" section to README** documenting the four ref forms, that large sets live on HF (`arcadia-impact/question-consistency-datasets`) and are auto-pulled+cached, the `scripts/upload_datasets_hf.py` command, and an example external use: `--items-path hf-dataset:Shengtao/recipe:train:title`. Commit:

```bash
git add README.md && git commit -m "docs: datasets resolution + HF hosting section"
```

- [ ] **Step 2: [OPS] Rename the GitHub repo + local remote**

```bash
gh repo rename question-consistency --yes
git remote set-url origin https://github.com/jonathanbostock/question-consistency.git
git remote -v   # confirm
```

---

## Task 9: Pull request [OPS]

- [ ] **Step 1: Push + open PR**

```bash
git push -u origin hf-datasets-and-rebrand
gh pr create --base main --title "HF dataset hosting + question-consistency rebrand" --body "<summary of rebrand, loader resolution, HF parquet hosting, removed large YAMLs, back-compat>"
```

---

## Self-Review

- **Spec coverage:** rebrand (Task 1), parquet HF configs (Tasks 5–6), footprint move + back-compat (Task 7), layered loader incl. external mode (Tasks 2–4), uploader xet-disabled (Task 5), dataset card (Task 6 Step 3), README (Task 8), repo rename (Task 8). All spec sections mapped.
- **Placeholders:** none — all code steps contain full code; the only `<…>` is the PR body summary (Task 9), an orchestrator free-text field.
- **Type consistency:** `resolve_items`, `read_yaml_items`, `_from_hf_config/_dataset/_file`, `HF_DATASET_REPO`, `REGISTRY`, `yaml_to_dataset` named consistently across Tasks 2–7.
- **Note for Codex:** Task 3 Step 2 flags the monkeypatch-interception caveat (import `datasets`/`huggingface_hub` as modules inside helpers so `monkeypatch.setattr` on the module attribute applies).
