from __future__ import annotations

import numpy as np

from .fit import predict_matrix_caseV


def decisiveness(mu) -> float:
    """mean|2 Phi - 1| over unordered pairs of the fitted Case V matrix. Bounded [0,1]."""
    P = predict_matrix_caseV(mu)
    iu = np.triu_indices(P.shape[0], k=1)
    return float(np.mean(np.abs(2 * P[iu] - 1)))


def decisiveness_raw(rows) -> float:
    """mean|2 p_util - 1| over observed edges (resolution-limited diagnostic)."""
    p = np.array([float(r["p_util"]) for r in rows])
    return float(np.mean(np.abs(2 * p - 1))) if len(p) else float("nan")


def transitivity_fas(rows, order) -> float:
    """1 - confidence-weighted fraction of edges pointing backward vs `order`
    (best->worst). Weight = |2 p_util - 1|."""
    rank = {item: r for r, item in enumerate(order)}
    num = den = 0.0
    for r in rows:
        i, j, p = r["i"], r["j"], float(r["p_util"])
        if i not in rank or j not in rank:
            continue
        w = abs(2 * p - 1)
        prefers_i = p > 0.5
        i_before_j = rank[i] < rank[j]   # i ranked better
        backward = prefers_i != i_before_j
        den += w
        if backward:
            num += w
    return 1.0 - (num / den) if den else float("nan")


def transitivity_triad(triads) -> float:
    """triads: list of (p_ab, p_bc, p_ca). Returns 1 - mean soft cycle mass."""
    if not triads:
        return float("nan")
    masses = []
    for p_ab, p_bc, p_ca in triads:
        fwd = p_ab * p_bc * p_ca
        bwd = (1 - p_ab) * (1 - p_bc) * (1 - p_ca)
        masses.append(fwd + bwd)
    return 1.0 - float(np.mean(masses))


def unidim_fit(mu, held_rows) -> dict:
    """Held-out Brier + log-loss of the fitted Case V model. Lower is better.
    noise_floor = irreducible Binomial variance for sample edges (else 0)."""
    P = predict_matrix_caseV(mu)
    briers, lls, floors = [], [], []
    for r in held_rows:
        y = float(r["p_util"])
        phat = float(np.clip(P[r["i"], r["j"]], 1e-9, 1 - 1e-9))
        briers.append((phat - y) ** 2)
        lls.append(-(y * np.log(phat) + (1 - y) * np.log(1 - phat)))
        if r.get("mode") == "sample":
            N = float(r["wins_i"] + r["wins_j"]) or 1.0
            floors.append(y * (1 - y) / N)
        else:
            floors.append(0.0)
    return {
        "brier": float(np.mean(briers)) if briers else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("nan"),
        "noise_floor": float(np.mean(floors)) if floors else 0.0,
    }


def reliability(reverse_pairs) -> dict:
    """reverse_pairs: list of {p_fwd, p_rev} where p_fwd=P(pick i|i first),
    p_rev=P(pick i|i second). order_consistency=1-mean|p_fwd+p_rev-1|,
    position_bias=mean(p_fwd+p_rev-1) (signed; >0 = first-slot preference)."""
    if not reverse_pairs:
        return {"order_consistency": float("nan"), "position_bias": float("nan")}
    diffs = np.array([p["p_fwd"] + p["p_rev"] - 1.0 for p in reverse_pairs])
    return {"order_consistency": float(1.0 - np.mean(np.abs(diffs))),
            "position_bias": float(np.mean(diffs))}


def question_robustness(cross_pairs) -> dict:
    """cross_pairs: list of {p_util_a, p_util_b} for the same item pair under two
    questions (both oriented to P(item_i > item_j))."""
    if not cross_pairs:
        return {"q_agreement": float("nan"), "q_sign_agreement": float("nan")}
    a = np.array([p["p_util_a"] for p in cross_pairs])
    b = np.array([p["p_util_b"] for p in cross_pairs])
    agree = 1.0 - np.mean(np.abs(a - b))
    sign = np.mean(np.sign(a - 0.5) == np.sign(b - 0.5))
    return {"q_agreement": float(agree), "q_sign_agreement": float(sign)}


from .fit import fit_caseV_mle, bootstrap_measurement, bootstrap_items


def _ci(samples, lo=2.5, hi=97.5):
    s = np.asarray([x for x in samples if np.isfinite(x)])
    if s.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(s, lo)), float(np.percentile(s, hi))]


def _bootstrap_raw(items, metric_fn, B, seed):
    """Percentile CI by resampling a list of observations with replacement."""
    rng = np.random.default_rng(seed)
    if not items:
        return [float("nan"), float("nan")]
    arr = np.array(items, dtype=object)
    vals = []
    for _ in range(B):
        sel = rng.integers(0, len(arr), size=len(arr))
        vals.append(metric_fn(list(arr[sel])))
    return _ci(vals)


def compute_panel(edges_by_phase, n, B=200, seed=0):
    elo = edges_by_phase.get("elo", [])
    res = fit_caseV_mle(elo, n=n, seed=seed)
    mu = res["mu"]
    order = list(np.argsort(-mu))           # best -> worst

    panel = {}

    # --- mu-derived metrics: measurement + generalization CIs via fit bootstraps ---
    panel["decisiveness"] = {
        "point": decisiveness(mu),
        "meas_ci": _ci(bootstrap_measurement(elo, n, B, decisiveness, seed)),
        "gen_ci": _ci(bootstrap_items(elo, n, max(B // 2, 30), decisiveness, seed)),
    }
    panel["decisiveness_raw"] = {"point": decisiveness_raw(elo),
                                 "meas_ci": _bootstrap_raw(elo, decisiveness_raw, B, seed),
                                 "gen_ci": [float("nan"), float("nan")]}

    def brier_of(mu_):
        return unidim_fit(mu_, elo)["brier"]
    panel["unidim_fit_brier"] = {
        "point": unidim_fit(mu, elo)["brier"],
        "meas_ci": _ci(bootstrap_measurement(elo, n, B, brier_of, seed)),
        "gen_ci": [float("nan"), float("nan")],
    }

    # --- raw-graph metrics: measurement CI by resampling their observation lists ---
    panel["transitivity_fas"] = {
        "point": transitivity_fas(elo, order),
        "meas_ci": _bootstrap_raw(elo, lambda r: transitivity_fas(r, order), B, seed),
        "gen_ci": [float("nan"), float("nan")],
    }
    triads = edges_by_phase.get("triad", [])
    panel["transitivity_triad"] = {
        "point": transitivity_triad(triads),
        "meas_ci": _bootstrap_raw(triads, transitivity_triad, B, seed),
        "gen_ci": [float("nan"), float("nan")],
    }
    rev = edges_by_phase.get("reverse", [])
    panel["order_consistency"] = {
        "point": reliability(rev)["order_consistency"],
        "meas_ci": _bootstrap_raw(rev, lambda r: reliability(r)["order_consistency"], B, seed),
        "gen_ci": [float("nan"), float("nan")],
    }
    cross = edges_by_phase.get("cross", [])
    panel["q_agreement"] = {
        "point": question_robustness(cross)["q_agreement"],
        "meas_ci": _bootstrap_raw(cross, lambda r: question_robustness(r)["q_agreement"], B, seed),
        "gen_ci": [float("nan"), float("nan")],
    }
    panel["mu_std_diagnostic"] = {"point": float(np.std(mu)),
                                  "meas_ci": [float("nan"), float("nan")],
                                  "gen_ci": [float("nan"), float("nan")]}
    return panel
