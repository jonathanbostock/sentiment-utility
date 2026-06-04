"""Pull two small-N "judgement" comparison datasets from HuggingFace and write them
as config/datasets/*.yaml (same schema as the existing item lists, plus a `meta`
block recording provenance + per-item tags).

  - leetcode_problems.yaml  <- greengerong/leetcode   (title + short gloss; Easy/Med/Hard)
  - recipes.yaml            <- Shengtao/recipe         (dish + short description; category)

Each item is a concise, self-contained one-liner so it fits inline in the pairwise
A/B template and even a 1b model can reason about it.

Run:  uv run python scripts/build_small_n_datasets.py
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "config" / "datasets"
TODAY = datetime.date.today().isoformat()

LEETCODE_SOURCE = "greengerong/leetcode"
RECIPE_SOURCE = "Shengtao/recipe"

# ---------------------------------------------------------------------------
# text cleaning
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")   # [text](url) -> text
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def clean_markdown(text: str) -> str:
    """Strip code fences / HTML / markdown links+emphasis and collapse whitespace."""
    text = _CODE_FENCE.sub(" ", text)
    text = _MD_IMG.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)               # keep link anchor text, drop the URL
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = _URL.sub(" ", text)                     # any bare URLs left over
    text = re.sub(r"[`*_#>\[\]]+", " ", text)       # markdown punctuation + stray brackets
    text = text.replace("\\", " ")
    text = _WS.sub(" ", text).strip()
    return text


def first_sentences(text: str, max_chars: int) -> str:
    """Return the leading whole sentence(s) up to max_chars (never cuts mid-sentence
    unless the first sentence alone is already too long, in which case hard-truncate)."""
    text = text.strip()
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not sent:
            continue
        if out and len(out) + 1 + len(sent) > max_chars:
            break
        out = sent if not out else f"{out} {sent}"
        if len(out) >= max_chars:
            break
    if not out:
        out = text
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return out


def make_item(title: str, gloss: str, max_chars: int) -> str:
    title = _WS.sub(" ", str(title)).strip().rstrip(".")
    gloss = first_sentences(clean_markdown(str(gloss)), max_chars)
    if not gloss:
        return title
    return f"{title}: {gloss}"


# ---------------------------------------------------------------------------
# dataset builders
# ---------------------------------------------------------------------------

def build_leetcode(n: int, seed: int, max_chars: int) -> tuple[list[str], list[dict]]:
    """Balanced across Easy/Medium/Hard. greengerong/leetcode is small (~2.3k rows)."""
    ds = load_dataset(LEETCODE_SOURCE, split="train")
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in ds:
        diff = (r.get("difficulty") or "").strip().title()
        content = r.get("content") or ""
        cleaned = clean_markdown(content)
        # drop premium-locked / empty / image-only statements
        if diff not in {"Easy", "Medium", "Hard"} or len(cleaned) < 60:
            continue
        if "subscribe to unlock" in cleaned.lower():
            continue
        by_diff[diff].append(r)

    rng = random.Random(seed)
    for v in by_diff.values():
        rng.shuffle(v)

    # near-equal quota per difficulty, summing to n
    diffs = ["Easy", "Medium", "Hard"]
    base, extra = divmod(n, len(diffs))
    quota = {d: base + (1 if k < extra else 0) for k, d in enumerate(diffs)}

    picked = {d: by_diff[d][: quota[d]] for d in diffs}
    # round-robin across difficulties so the YAML isn't blocked by level (purely cosmetic;
    # the sampler ignores list order and the fit remaps indices)
    items, meta = [], []
    for k in range(max(quota.values())):
        for d in diffs:
            if k < len(picked[d]):
                r = picked[d][k]
                items.append(make_item(r["title"], r.get("content", ""), max_chars))
                meta.append({"difficulty": d, "slug": r.get("slug"), "qid": r.get("id")})
    return items, meta


def build_recipes(n: int, seed: int, max_chars: int, scan: int = 12000) -> tuple[list[str], list[dict]]:
    """Varied across food categories. Stream up to `scan` rows (caps the download)."""
    ds = load_dataset(RECIPE_SOURCE, split="train", streaming=True)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    seen_titles = set()
    for r in itertools.islice(ds, scan):
        title = (r.get("title") or "").strip()
        desc = (r.get("description") or "").strip()
        cat = (r.get("category") or "other").strip() or "other"
        key = title.lower()
        if not title or not desc or len(desc) < 40 or key in seen_titles:
            continue
        seen_titles.add(key)
        by_cat[cat].append(r)

    rng = random.Random(seed)
    cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))   # biggest categories first
    for c in cats:
        rng.shuffle(by_cat[c])

    # round-robin across categories for variety
    items, meta = [], []
    cursors = {c: 0 for c in cats}
    while len(items) < n and any(cursors[c] < len(by_cat[c]) for c in cats):
        for c in cats:
            if len(items) >= n:
                break
            if cursors[c] < len(by_cat[c]):
                r = by_cat[c][cursors[c]]
                cursors[c] += 1
                item = make_item(r["title"], r.get("description", ""), max_chars)
                if len(item) < 12:
                    continue
                items.append(item)
                meta.append({"category": c, "total_time": r.get("total_time"),
                             "rating": r.get("rating")})
    return items, meta


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def write_yaml(path: Path, dataset: str, source: str, items: list[str], meta: list[dict]):
    payload = {
        "meta": {
            "dataset": dataset,
            "source": f"huggingface:{source}",
            "built": TODAY,
            "n": len(items),
            "note": "small-N pairwise-judgement dataset; per_item is index-aligned with items",
            "per_item": meta,
        },
        "items": items,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {len(items):2d} items -> {path.relative_to(REPO)}  (source {source})")
    for it in items[:4]:
        print(f"    e.g. {it}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=170, help="max gloss length per item")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== leetcode ({LEETCODE_SOURCE}) ==")
    lc_items, lc_meta = build_leetcode(args.n, args.seed, args.max_chars)
    write_yaml(OUT_DIR / "leetcode_problems.yaml", "leetcode_problems", LEETCODE_SOURCE,
               lc_items, lc_meta)

    print(f"== recipes ({RECIPE_SOURCE}) ==")
    rc_items, rc_meta = build_recipes(args.n, args.seed, args.max_chars)
    write_yaml(OUT_DIR / "recipes.yaml", "recipes", RECIPE_SOURCE, rc_items, rc_meta)


if __name__ == "__main__":
    main()
