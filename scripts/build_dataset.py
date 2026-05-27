from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

import yaml

from sentiment_utility.dataset import build_pool_sample


WARRINER_URL = "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv"
THINGS_URLS = [
    "https://raw.githubusercontent.com/ViCCo-Group/THINGSplus/main/data/things_concepts.tsv",
    "https://raw.githubusercontent.com/ViCCo-Group/THINGSplus/main/data/THINGS_concepts.tsv",
    "https://raw.githubusercontent.com/ThomasHebart/things-database/main/things_concepts.tsv",
]


def _fetch_text(url: str, timeout: float = 20.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8-sig")


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def load_curated(path: Path) -> list[tuple[str, None]]:
    data = yaml.safe_load(path.read_text()) or {}
    concepts = data.get("concepts", [])
    return [(str(concept), None) for concept in concepts]


def fetch_warriner() -> list[tuple[str, float | None]]:
    try:
        text = _fetch_text(WARRINER_URL)
        rows = csv.DictReader(io.StringIO(text))
        out = []
        for row in rows:
            word = (row.get("Word") or "").strip()
            if not word:
                continue
            valence_text = (row.get("V.Mean.Sum") or "").strip()
            try:
                valence = float(valence_text)
            except ValueError:
                valence = None
            out.append((word, valence))
        return out
    except Exception as exc:
        _warn(f"failed to fetch Warriner CSV: {exc}")
        return []


def fetch_things() -> list[tuple[str, None]]:
    errors = []
    for url in THINGS_URLS:
        try:
            text = _fetch_text(url)
            rows = csv.DictReader(io.StringIO(text), delimiter="\t")
            if rows.fieldnames is None:
                raise ValueError("missing header row")
            fields = {field.lower(): field for field in rows.fieldnames}
            concept_field = fields.get("concept") or fields.get("word")
            if concept_field is None:
                raise ValueError(f"missing concept/Word column in {rows.fieldnames}")
            concepts = []
            for row in rows:
                concept = (row.get(concept_field) or "").strip()
                if concept:
                    concepts.append((concept, None))
            if concepts:
                return concepts
            raise ValueError("no concepts found")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    _warn("failed to fetch THINGS concept list: " + " | ".join(errors))
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blended concept dataset.")
    parser.add_argument("--n", type=int, default=500, help="Number of items to write.")
    parser.add_argument(
        "--out",
        default="config/items_500.yaml",
        help="Output YAML path.",
    )
    parser.add_argument(
        "--curated-quota",
        type=int,
        default=250,
        help="Target number of curated concepts.",
    )
    parser.add_argument(
        "--things-quota",
        type=int,
        default=150,
        help="Target number of THINGS concepts.",
    )
    parser.add_argument(
        "--warriner-quota",
        type=int,
        default=None,
        help="Target number of Warriner concepts. Defaults to n - curated quota.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warriner_quota = (
        args.n - args.curated_quota if args.warriner_quota is None else args.warriner_quota
    )
    curated = load_curated(Path("config/curated_concepts.yaml"))
    sources = {
        "curated": curated,
        "things": fetch_things(),
        "warriner": fetch_warriner(),
    }
    # THINGS has no stable raw URL (OSF-only), so it typically resolves empty;
    # curated (rich concepts) + Warriner (words) fill to 500. If THINGS is
    # available it contributes and Warriner's quota tops up the remainder.
    items, meta = build_pool_sample(
        sources,
        quotas={
            "curated": args.curated_quota,
            "things": args.things_quota,
            "warriner": warriner_quota,
        },
        n=args.n,
        seed=0,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({"items": items, "meta": meta}, sort_keys=False))

    counts = {source: 0 for source in sources}
    for item in items:
        counts[meta[item]["source"]] += 1
    print(f"wrote {output} ({len(items)} items)")
    for source, count in counts.items():
        print(f"{source}: {count}")


if __name__ == "__main__":
    main()
