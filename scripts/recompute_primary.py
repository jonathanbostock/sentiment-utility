"""Recompute the OCT panel with mu fit on the PRIMARY (+1) question only, from the saved
edges.jsonl (no GPU/re-run). Writes panel_primary.json per model and prints the updated
persona table + correlation/PCA factor analysis.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from question_consistency.panel import compute_panel


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load("scripts/build_coherence.py", "bc")
af = _load("scripts/analyze_panel_factor.py", "af")
N = len(yaml.safe_load(open("config/datasets/items_2000.yaml"))["items"])

rows = {}
for ej in sorted(Path("runs/oct2k").glob("*/edges.jsonl")):
    model = ej.parent.name
    edges = [json.loads(l) for l in open(ej)]
    panel = compute_panel(bc._bucket(edges), n=N, primary_qid="pos", fit_steps=1200)
    (ej.parent / "panel_primary.json").write_text(json.dumps(panel, indent=2))
    rows[model] = {m: panel.get(m, {}).get("point", float("nan")) for m in af.METRIC_SIGN}

df = pd.DataFrame(rows).T[list(af.METRIC_SIGN)]
print("=== primary-only panel point estimates ===")
print(df.round(3).to_string())

pers = df.drop(index="base")
print("\n=== decisiveness ranking (primary-only) ===")
print(df["decisiveness"].sort_values(ascending=False).round(3).to_string())

aligned = af.sign_align(df).dropna(axis=1, how="all").dropna(axis=0, how="any")
pear = aligned.corr(method="pearson")
print("\n=== Pearson correlation (primary-only, sign-aligned) ===")
print(pear.round(2).to_string())
var, Vt = af.pca_numpy(aligned.values)
print("\nPCA variance explained: " + ", ".join(f"PC{k+1}={v:.2f}" for k, v in enumerate(var[:4])))
print(f"PC1 = {var[0]*100:.1f}%")
