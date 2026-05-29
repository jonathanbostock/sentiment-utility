"""Quantify how much pooling the +1/-1 question framings deflates decisiveness.

Refits Case V mu on each model's elo edges three ways -- pooled (both questions),
pos-only (+1), neg-only (-1) -- and reports decisiveness for each. If pos-only >> pooled,
framing disagreement is deflating the pooled decisiveness. Uses saved edges.jsonl (no GPU).
"""
import json
import yaml
from sentiment_utility.fit import fit_caseV_mle
from sentiment_utility.panel import decisiveness

N = len(yaml.safe_load(open("config/items_2000.yaml"))["items"])


def elo_rows(model):
    out = []
    for line in open(f"runs/oct2k/{model}/edges.jsonl"):
        r = json.loads(line)
        if r.get("phase", "elo") == "elo":
            out.append(r)
    return out


def main():
    print(f"{'model':12s} {'pooled':>7s} {'pos_only':>8s} {'neg_only':>8s}  (n_pos/n_neg)")
    for m in ["base", "sarcasm", "goodness", "poeticism", "mathematical"]:
        elo = elo_rows(m)
        pos = [r for r in elo if r.get("question_id") == "pos"]
        neg = [r for r in elo if r.get("question_id") == "neg"]
        da = decisiveness(fit_caseV_mle(elo, n=N, steps=1000)["mu"])
        dp = decisiveness(fit_caseV_mle(pos, n=N, steps=1000)["mu"]) if pos else float("nan")
        dn = decisiveness(fit_caseV_mle(neg, n=N, steps=1000)["mu"]) if neg else float("nan")
        print(f"{m:12s} {da:7.3f} {dp:8.3f} {dn:8.3f}  ({len(pos)}/{len(neg)})")


if __name__ == "__main__":
    main()
