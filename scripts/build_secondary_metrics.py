"""Build coherence_four_metrics-style CSVs for the SECONDARY question banks (interest, shape).

Same pipeline as build_four_metrics.py (raw edges -> four_metrics + decis_mu, attach
MMLU + fitted ECI), but over the Gemma/Qwen/Llama scale series re-run on each secondary
bank. Reads the retrieved tarballs in results/series_runs/secondary/ plus the salvaged
Qwen runs in /tmp/qwen_salvage/. Writes results/coherence_secondary_<q>.csv per question.

Models that are not present yet (e.g. a still-running Qwen-72B) are simply skipped, so this
can be run mid-flight and re-run when the rest land.
"""
import sys
import tarfile
import tempfile
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import metrics_cached, FOUR          # noqa: E402
from build_four_metrics import MMLU, parse_b, _eci_key  # noqa: E402

SECONDARY = REPO / "results/series_runs/secondary"
QWEN_SALVAGE = Path("/tmp/qwen_salvage")
CLAUDE_DIR = SECONDARY / "claude45_secondary"   # extracted Claude-4.5 API runs (haiku/sonnet)
QUESTIONS = ["interest", "shape"]


def _family(name: str) -> str:
    n = name.lower()
    if n.startswith("gemma"):
        return "Gemma"
    if n.startswith("qwen"):
        return "Qwen"
    if n.startswith("claude"):
        return "Claude 4.5"
    return "Llama"


def _mmlu_key(raw: str) -> str:
    # MMLU dict keys: gemma without "-it"; llama/qwen keep "-instruct".
    return raw.replace("-it", "")


def _collect(q: str, tmp: Path):
    """Yield (family, raw_model, edges_path) for question q from all sources present."""
    # Gemma + Llama come from the retrieved tarballs.
    for fam in ("gemma", "llama"):
        tb = SECONDARY / f"{fam}-{q}.tar.gz"
        if tb.exists():
            with tarfile.open(tb) as t:
                t.extractall(tmp / f"{fam}-{q}")
            for r in sorted((tmp / f"{fam}-{q}" / "runs/elicit").glob(f"*--{q}")):
                edges = r / "edges.jsonl"
                if edges.exists() and edges.stat().st_size > 0:
                    yield _family(r.name), r.name[: -len(f"--{q}")], edges
    # Qwen comes from the salvaged dir (+ the 72B redo once it lands there).
    qd = QWEN_SALVAGE / q
    if qd.exists():
        for r in sorted(qd.glob(f"qwen*--{q}")):
            edges = r / "edges.jsonl"
            if edges.exists() and edges.stat().st_size > 0:
                yield "Qwen", r.name[: -len(f"--{q}")], edges
    # Claude 4.5 (haiku/sonnet) from the API run, extracted under CLAUDE_DIR.
    if CLAUDE_DIR.exists():
        for r in sorted(CLAUDE_DIR.glob(f"claude*--{q}")):
            edges = r / "edges.jsonl"
            if edges.exists() and edges.stat().st_size > 0:
                yield "Claude 4.5", r.name[: -len(f"--{q}")], edges


def build(q: str) -> pd.DataFrame:
    recs = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fam, raw, edges in _collect(q, tmp):
            recs.append({"family": fam, "model": raw, "params_b": parse_b(raw),
                         **metrics_cached(edges)})
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["mmlu"] = df["model"].map(lambda m: MMLU.get(_mmlu_key(m)))
    eci = pd.read_csv(REPO / "results/eci_scores.csv")
    ECI = {_eci_key(m): v for m, v in zip(eci["model"], eci["eci"])}
    df["eci"] = df["model"].map(lambda m: ECI.get(_eci_key(m)))
    df = df.sort_values(["family", "eci"])
    return df[["family", "model", "params_b", "mmlu", "eci", "decis_mu", "fit_r2"] + FOUR]


def main():
    for q in QUESTIONS:
        df = build(q)
        if df.empty:
            print(f"[{q}] no runs found yet — skipped")
            continue
        out = REPO / f"results/coherence_secondary_{q}.csv"
        df.to_csv(out, index=False)
        print(f"=== {q} ({len(df)} models) ===")
        print(df.round(3).to_string(index=False))
        print(f"wrote {out}\n")


if __name__ == "__main__":
    main()
