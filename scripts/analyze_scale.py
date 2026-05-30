"""Scaling analysis for the model-series sweep: read each series' panel.json + mu.json from
the pulled tarballs, map each run to its scale axis (params for Qwen/Llama; pretraining tokens
for OLMo), compute mu_valence_corr against Warriner norms, write a combined CSV + seaborn PDFs.

Idempotent: re-run as more series land. Reads results/series_runs/<series>/<series>_*.tar.gz.
"""
from __future__ import annotations
import json, re, glob, tarfile, csv
from pathlib import Path
import numpy as np

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
SR = REPO / "results/series_runs"
VAL = json.load(open("/tmp/items2000_valence.json")) if Path("/tmp/items2000_valence.json").exists() else {}
PANEL_METRICS = ["decisiveness", "transitivity_fas", "transitivity_triad",
                 "order_consistency", "q_agreement", "unidim_fit_log_loss", "mu_std_diagnostic"]

def pt(panel, k):
    d = panel.get(k, {}); return d.get("point") if isinstance(d, dict) else d

def mu_valence_corr(mu_path):
    if not VAL: return float("nan")
    mu = json.load(open(mu_path))
    xs, ys = [], []
    for item, m in mu.items():
        if item in VAL:
            xs.append(m); ys.append(VAL[item])
    if len(xs) < 10 or np.std(xs) < 1e-9: return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])

def parse_scale(series, name):
    """Return (x_value, x_kind, label)."""
    if series == "olmo":
        m = re.search(r"tokens?(\d+)B|(\d+)Btok", name)
        if m:
            tok = int(m.group(1) or m.group(2)); return tok, "tokens_B", f"{tok}B"
        if "base-final" in name: return 3896, "tokens_B", "final-base"   # end of stage1 anneal
        if "instruct" in name: return None, "posttrained", "Instruct"
        return None, "?", name
    # qwen / llama: params in billions from name like '...-32b-...' or '...-0.5b-...'
    m = re.search(r"(\d+\.?\d*)b", name.lower())
    if m: p = float(m.group(1)); return p, "params_B", f"{p:g}B"
    return None, "?", name

def load_series():
    rows = []
    for series in ["qwen", "llama", "olmo", "big"]:
        for tb in glob.glob(str(SR / series / f"{series}_*.tar.gz")):
            ext = SR / series / "x"; ext.mkdir(exist_ok=True)
            with tarfile.open(tb) as t: t.extractall(ext, filter="data")
            for pj in glob.glob(str(ext / "**/panel.json"), recursive=True):
                pj = Path(pj); name = pj.parent.name
                panel = json.loads(pj.read_text())
                mp = pj.parent / "mu.json"
                x, kind, lab = parse_scale(series, name)
                # 'big' (70B/72B) belongs to whichever family by name
                fam = series
                if series == "big":
                    fam = "llama" if "llama" in name else "qwen" if "qwen" in name else "big"
                row = {"series": fam, "run": name, "x": x, "x_kind": kind, "label": lab,
                       "mu_valence_corr": round(mu_valence_corr(mp), 4) if mp.exists() else float("nan")}
                for m in PANEL_METRICS:
                    v = pt(panel, m); row[m] = round(v, 4) if v is not None else None
                rows.append(row)
    return rows

def main():
    rows = load_series()
    if not rows:
        print("no series data yet"); return
    out = REPO / "results/coherence_scale_all.csv"
    keys = ["series", "run", "label", "x", "x_kind", "mu_valence_corr"] + PANEL_METRICS
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in sorted(rows, key=lambda r: (r["series"], r["x"] or 0)): w.writerow(r)
    print(f"wrote {out} ({len(rows)} runs)")
    for r in sorted(rows, key=lambda r: (r["series"], r["x"] or 0)):
        print(f"  {r['series']:6s} {r['label']:12s} dec={r['decisiveness']} q_corr={r['q_agreement']} "
              f"order={r['order_consistency']} val_corr={r['mu_valence_corr']}")

if __name__ == "__main__":
    main()
