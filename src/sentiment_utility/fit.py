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
