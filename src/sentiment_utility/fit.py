from __future__ import annotations

import math
import numpy as np
import torch

_SQRT2 = math.sqrt(2.0)


def _phi(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def normalize_edges(rows):
    """rows -> (i_idx, j_idx, w_pos, w_neg) as numpy arrays. See contract in plan."""
    i_idx, j_idx, w_pos, w_neg = [], [], [], []
    for r in rows:
        i_idx.append(int(r["i"]))
        j_idx.append(int(r["j"]))
        if r.get("mode") == "sample":
            w_pos.append(float(r["wins_i"]))
            w_neg.append(float(r["wins_j"]))
        else:
            p = float(r["p_util"])
            w_pos.append(p)
            w_neg.append(1.0 - p)
    return (np.asarray(i_idx), np.asarray(j_idx),
            np.asarray(w_pos, dtype=np.float64), np.asarray(w_neg, dtype=np.float64))


def predict_matrix_caseV(mu) -> np.ndarray:
    mu_t = torch.as_tensor(np.asarray(mu), dtype=torch.float64)
    diff = mu_t[:, None] - mu_t[None, :]
    P = _phi(diff / _SQRT2)              # P_ij = Phi((mu_i-mu_j)/sqrt2), Case V with sigma=1
    P.fill_diagonal_(0.5)
    return P.numpy()


def fit_caseV_mle(rows, n, steps=2000, lr=0.05, seed=0, device=None) -> dict:
    """Homoscedastic Thurstone Case V MLE on mu (sigma fixed=1). No prior.
    P(item_i > item_j) = Phi((mu_i - mu_j)/sqrt2). Gauge: center mu."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    ii = torch.as_tensor(i_idx, device=device)
    jj = torch.as_tensor(j_idx, device=device)
    wp = torch.as_tensor(w_pos, device=device)
    wn = torch.as_tensor(w_neg, device=device)
    mu = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        P = _phi((mu[ii] - mu[jj]) / _SQRT2).clamp(1e-9, 1 - 1e-9)
        nll = -(wp * torch.log(P) + wn * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu_c = mu - mu.mean()            # additive gauge
    return {"mu": mu_c.detach().cpu().numpy()}


def bootstrap_measurement(rows, n, B, metric_fn, seed=0, steps=1500, lr=0.05, device=None):
    """B MLE refits at once as one (B, n) mu tensor. Resamples edges with replacement
    per replicate; for sample edges also resamples wins ~ Binomial(N, wins_i/N)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    is_sample = np.array([r.get("mode") == "sample" for r in rows])
    totals = w_pos + w_neg                            # N for sample, 1.0 for soft
    E = len(rows)

    pick = rng.integers(0, E, size=(B, E))            # (B, E) per-replicate edge resample
    wp = w_pos[pick].copy()                           # (B, E)
    wn = w_neg[pick].copy()
    smask = is_sample[pick]                           # (B, E) bool
    if smask.any():
        Ns = totals[pick]
        ps = np.where(Ns > 0, wp / np.maximum(Ns, 1e-9), 0.5)
        draws = rng.binomial(np.where(smask, Ns, 0).astype(int), np.clip(ps, 0, 1))
        wp = np.where(smask, draws, wp)
        wn = np.where(smask, Ns - draws, wn)

    ii_rep = torch.as_tensor(i_idx[pick], device=device)        # (B, E)
    jj_rep = torch.as_tensor(j_idx[pick], device=device)
    wp_t = torch.as_tensor(wp, device=device, dtype=torch.float64)
    wn_t = torch.as_tensor(wn, device=device, dtype=torch.float64)

    mu = torch.zeros(B, n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=lr)
    bidx = torch.arange(B, device=device)[:, None]
    for _ in range(steps):
        opt.zero_grad()
        diff = (mu[bidx, ii_rep] - mu[bidx, jj_rep]) / _SQRT2
        P = _phi(diff).clamp(1e-9, 1 - 1e-9)
        nll = -(wp_t * torch.log(P) + wn_t * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu_c = (mu - mu.mean(dim=1, keepdim=True)).detach().cpu().numpy()
    return np.array([metric_fn(mu_c[b]) for b in range(B)])


def bootstrap_items(rows, n, B, metric_fn, seed=0, steps=1500, lr=0.05):
    """Cluster bootstrap over items: resample item ids, refit induced sub-graph."""
    rng = np.random.default_rng(seed)
    out = []
    by_pair = rows
    for b in range(B):
        keep = rng.integers(0, n, size=n)            # resampled item ids (with dups)
        remap = {old: new for new, old in enumerate(keep)}
        sub = [dict(r, i=remap[r["i"]], j=remap[r["j"]])
               for r in by_pair if r["i"] in remap and r["j"] in remap]
        if not sub:
            continue
        res = fit_caseV_mle(sub, n=len(keep), steps=steps, lr=lr, seed=b)
        out.append(metric_fn(res["mu"]))
    return np.asarray(out)
