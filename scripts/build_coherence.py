from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from question_consistency.io_utils import load_items
from question_consistency.panel import compute_panel


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
    import argparse
    ap = argparse.ArgumentParser(description="Emit a panel CSV from a directory of runs.")
    ap.add_argument("--runs-dir", default="runs/oct2k",
                    help="dir of <model>/edges.jsonl runs to score")
    ap.add_argument("--items-path", default="config/datasets/items_2000.yaml")
    ap.add_argument("--primary-qid", default="pos",
                    help="fit mu on this question_id only (None to pool all)")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--out", default="results/coherence_all_v5.csv")
    args = ap.parse_args()
    primary = None if args.primary_qid in ("", "none", "None") else args.primary_qid

    runs = sorted(p.parent for p in Path(args.runs_dir).glob("*/edges.jsonl"))
    rows = []
    for run in runs:
        row = {"model": run.name}
        row.update(panel_row_from_edges(run / "edges.jsonl", args.items_path,
                                        bootstrap=args.bootstrap, B=args.B, primary_qid=primary))
        rows.append(row)
        print(f"  {run.name:14s} decisiveness={row['decisiveness_point']:.3f} "
              f"transitivity_fas={row['transitivity_fas_point']:.3f} "
              f"order_consistency={row['order_consistency_point']:.3f} "
              f"q_agreement={row['q_agreement_point']:.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model"] + [k for k in rows[0] if k != "model"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {out}  (primary_qid={primary})")


if __name__ == "__main__":
    main()
