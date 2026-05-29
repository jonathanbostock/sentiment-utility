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
    # default (bootstrap off): finite point, NaN CIs
    row = mod.panel_row_from_edges(edges_path, items_path)
    assert 0.0 <= row["decisiveness_point"] <= 1.0
    assert math.isnan(row["decisiveness_meas_lo"])
    # bootstrap on: CI brackets the point
    row_b = mod.panel_row_from_edges(edges_path, items_path, bootstrap=True, B=40)
    assert row_b["decisiveness_meas_lo"] <= row_b["decisiveness_point"] <= row_b["decisiveness_meas_hi"]
