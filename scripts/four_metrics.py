"""The four agreement-probability consistency metrics, computed from raw edges.

All four are "probability that two draws agree", from the per-comparison choice probability:
  p_self     P(same answer if Q(A,B) sampled twice)         = E[p^2+(1-p)^2]      (elo phase)
  p_reversal P(same concept under Q(A,B) vs Q(B,A))          = E[pf(1-pr)+(1-pf)pr] (reverse)
             where pf=P(pick i|i first), pr=P(pick j|j first)
  p_acyclic  P(no cycle over Q(A,B),Q(B,C),Q(C,A))           = 1 - E[cycle mass]   (triad)
  p_crossq   P(same answer under Q_i(A,B) vs Q_j(A,B))       = E[pa*pb+(1-pa)(1-pb)] (cross)

Chance floors (indifferent model): p_self/p_reversal/p_crossq = 0.5, p_acyclic = 0.75.
p_self is the decisiveness/determinism baseline; read p_reversal & p_crossq relative to it
(p_self - p_reversal = order-bias cost; p_self - p_crossq = framing cost).

Sample-mode refinement: p_self reuses the SAME comparison twice, so the plug-in p_hat^2+... is
biased at small n. When raw wins are present we use the unbiased U-statistic
[k(k-1)+(n-k)(n-k-1)]/[n(n-1)]. p_reversal/p_acyclic/p_crossq multiply INDEPENDENT comparisons,
so plug-in is already unbiased there.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
sys.path.insert(0, str(REPO / "scripts"))
from build_coherence import _bucket, _load_edges

FOUR = ["p_self", "p_reversal", "p_acyclic", "p_crossq"]
LAB = {
    "p_self":     "P(repeat)  ·  self-agreement",
    "p_reversal": "P(agree | A,B vs B,A)  ·  order",
    "p_acyclic":  "P(no cycle)  ·  transitivity",
    "p_crossq":   "P(agree | Qᵢ vs Qⱼ)  ·  framing",
}


def _unsmooth(edges):
    """Sample-mode p_util/p_a are Jeffreys-smoothed (k+0.5)/(n+1), which biases the
    multilinear metrics (p_reversal/p_crossq deflated toward 0.5, p_acyclic inflated by
    damping observed cycles). Replace with raw k/n — unbiased for these independent-comparison
    products even at small n. Logprob edges (no wins) pass through unchanged."""
    out = []
    for e in edges:
        wi, wj = e.get("wins_i"), e.get("wins_j")
        if wi is None:
            out.append(e); continue
        n = wi + wj
        if n == 0:
            continue                       # no parsed samples -> no information
        e = dict(e)
        e["p_util"] = wi / n
        if e.get("p_a") is not None:
            e["p_a"] = min(1.0, max(0.0, (e["p_a"] * (n + 1) - 0.5) / n))   # invert Jeffreys
        out.append(e)
    return out


def compute_four(edges_path) -> dict:
    raw = _unsmooth(_load_edges(edges_path))
    b = _bucket(raw)
    elo = b["elo"]

    # p_self: unbiased U-statistic from wins if sample-mode, else analytic from p_util
    us = []
    for e in elo:
        ki, kj = e.get("wins_i"), e.get("wins_j")
        if ki is None:
            break
        n = ki + kj
        if n >= 2:
            us.append((ki * (ki - 1) + kj * (kj - 1)) / (n * (n - 1)))
    if us:
        p_self = float(np.mean(us))
    else:
        p = np.array([e["p_util"] for e in elo])
        p_self = float(np.mean(p ** 2 + (1 - p) ** 2))

    rv = b["reverse"]
    pf = np.array([r["p_fwd"] for r in rv]); pr = np.array([r["p_rev"] for r in rv])
    p_reversal = float(np.mean(pf * (1 - pr) + (1 - pf) * pr))

    tr = b["triad"]
    p_acyclic = float(1 - np.mean([a * bb * c + (1 - a) * (1 - bb) * (1 - c) for a, bb, c in tr]))

    cr = b["cross"]
    pa = np.array([c["p_util_a"] for c in cr]); pb = np.array([c["p_util_b"] for c in cr])
    p_crossq = float(np.mean(pa * pb + (1 - pa) * (1 - pb)))

    return {"p_self": p_self, "p_reversal": p_reversal,
            "p_acyclic": p_acyclic, "p_crossq": p_crossq}
