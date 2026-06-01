"""Recompute unidim_fit_brier (and verify the other metrics) for the Qwen2.5 + Llama-3.x
scale-sweep models from their saved edges, because coherence_scale_all.csv predates the Brier
column. Edges live in the series_runs tarballs; we extract, fit the panel with current
(corr-form) code, and emit results/coherence_scale_brier.csv keyed by run.

Run: uv run python scripts/build_scale_brier.py
"""
import sys
import tarfile
import tempfile
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_coherence import panel_row_from_edges

TARBALLS = [
    REPO / "results/series_runs/llama/llama_20260530.tar.gz",
    REPO / "results/series_runs/qwen/qwen_20260530.tar.gz",
    REPO / "results/series_runs/big/big2_20260530.tar.gz",   # llama-70b + qwen-72b
]
ITEMS = REPO / "config/items_2000.yaml"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for tb in TARBALLS:
            with tarfile.open(tb) as t:
                t.extractall(tmp)
        runs = sorted(p.parent for p in (tmp / "runs/elicit").glob("*-instruct/edges.jsonl")
                      if p.parent.name.startswith(("qwen", "llama")))
        rows = []
        for r in runs:
            flat = panel_row_from_edges(r / "edges.jsonl", ITEMS)
            rows.append({"run": r.name,
                         "unidim_fit_brier": round(flat["unidim_fit_brier_point"], 4),
                         "q_check": round(flat["q_agreement_point"], 3),
                         "ll_check": round(flat["unidim_fit_log_loss_point"], 4)})
            print(rows[-1])
    out = REPO / "results/coherence_scale_brier.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
