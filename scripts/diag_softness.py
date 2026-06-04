"""Diagnose what still deflates decisiveness after removing framing pooling.

For each model's PRIMARY-question (+1) elo edges:
  - raw p_util distribution (is the forced-choice readout saturated or soft?)
  - raw decisiveness vs fitted decisiveness
  - fitted mu_std and fraction of all item-pairs the fit calls decisive (|2Phi-1|>0.8)
"""
import json
import numpy as np
import yaml
from question_consistency.fit import fit_caseV_mle, predict_matrix_caseV

N = len(yaml.safe_load(open("config/datasets/items_2000.yaml"))["items"])


def pos_elo(model):
    rows = []
    for line in open(f"runs/oct2k/{model}/edges.jsonl"):
        r = json.loads(line)
        if r.get("phase", "elo") == "elo" and r.get("question_id") == "pos":
            rows.append(r)
    return rows


def main():
    for m in ["base", "sarcasm", "poeticism"]:
        rows = pos_elo(m)
        p = np.array([r["p_util"] for r in rows])
        praw = np.abs(2 * p - 1)
        mu = fit_caseV_mle(rows, n=N, steps=1500)["mu"]
        P = predict_matrix_caseV(mu)
        iu = np.triu_indices(N, k=1)
        fit_dec = float(np.mean(np.abs(2 * P[iu] - 1)))
        print(f"\n=== {m} (pos-only, {len(rows)} elo edges) ===")
        print(f"  raw p_util:   min={p.min():.3f} p10={np.percentile(p,10):.3f} "
              f"med={np.median(p):.3f} p90={np.percentile(p,90):.3f} max={p.max():.3f}")
        print(f"  raw |2p-1|:   mean={praw.mean():.3f} (=raw decisiveness)  "
              f"frac>0.8={np.mean(praw>0.8):.3f}  frac>0.6={np.mean(praw>0.6):.3f}")
        print(f"  fitted:       decisiveness={fit_dec:.3f}  mu_std={mu.std():.3f}  "
              f"frac pairs |2Phi-1|>0.8 = {np.mean(np.abs(2*P[iu]-1)>0.8):.3f}")


if __name__ == "__main__":
    main()
