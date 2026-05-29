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
    cross, triad_putil = [], []
    rev_pa = {}           # (i,j) -> {orientation: raw P(pick slot-A)}
    for r in rows:
        ph = r.get("phase")
        if ph == "reverse":
            rev_pa.setdefault((r["i"], r["j"]), {})[r.get("orientation")] = r.get("p_a")
        elif ph == "triad":
            triad_putil.append(r["p_util"])
        elif ph == "cross_question" and (r["i"], r["j"]) in fwd:
            cross.append({"p_util_a": fwd[(r["i"], r["j"])], "p_util_b": r["p_util"]})
    reverse = [{"p_fwd": v["i"], "p_rev": v["j"]} for v in rev_pa.values()
               if v.get("i") is not None and v.get("j") is not None]
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


def panel_row_from_edges(edges_path, items_path, bootstrap=False, B=200, primary_qid=None):
    items = load_items(items_path)
    rows = _load_edges(edges_path)
    panel = compute_panel(_bucket(rows), n=len(items), bootstrap=bootstrap, B=B,
                          primary_qid=primary_qid)
    return _flatten(panel)


def main():
    # Registry of (group, model, family, role, edges_path, items_path); populate after the
    # weekend re-elicitation, pointing at each run's edges.jsonl, then write
    # results/coherence_all_v5.csv via panel_row_from_edges + csv.DictWriter.
    raise SystemExit("populate the run registry then write results/coherence_all_v5.csv")


if __name__ == "__main__":
    main()
