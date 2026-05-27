from __future__ import annotations

import numpy as np
import torch

from .thurstone import _phi, predict_pref_matrix


def rank_by_quicksort(n, oracle, seed=0):
    """Randomized batched-pivot quicksort. Returns order best->worst and edges."""
    rng = np.random.default_rng(seed)
    edges = []

    def sort(bucket):
        if len(bucket) <= 1:
            return list(bucket)
        pivot = bucket[rng.integers(len(bucket))]
        rest = [x for x in bucket if x != pivot]
        pairs = [(x, pivot) for x in rest]
        probs = oracle(pairs)
        greater, lesser = [], []
        for x in rest:
            p = probs[(x, pivot)]
            edges.append((x, pivot, float(p)))
            (greater if p > 0.5 else lesser).append(x)
        return sort(greater) + [pivot] + sort(lesser)

    order = sort(list(range(n)))
    return order, edges


def spacing_pass(order, oracle, k=2):
    pairs = []
    for r in range(len(order)):
        for d in range(1, k + 1):
            if r + d < len(order):
                pairs.append((order[r], order[r + d]))
    probs = oracle(pairs)
    return [(i, j, float(probs[(i, j)])) for (i, j) in pairs]


def edges_to_implied_matrix(mu, sigma):
    return predict_pref_matrix(mu, sigma)


def fit_thurstone_sparse(
    edges, n, lr=0.05, steps=2000, test_frac=0.2, l2_sigma=0.01, seed=0
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = len(edges)
    perm = torch.randperm(m, generator=g).tolist()
    n_test = int(test_frac * m)
    test_set = set(perm[:n_test])

    ii = torch.tensor([e[0] for e in edges], device=device)
    jj = torch.tensor([e[1] for e in edges], device=device)
    pp = torch.tensor([e[2] for e in edges], dtype=torch.float64, device=device)
    is_train = torch.tensor([k not in test_set for k in range(m)], device=device)

    mu = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    log_sigma = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        sigma = torch.exp(log_sigma)
        denom = torch.sqrt(sigma[ii] ** 2 + sigma[jj] ** 2)
        p = _phi((mu[ii] - mu[jj]) / denom).clamp(1e-6, 1 - 1e-6)
        bce = -(pp * torch.log(p) + (1 - pp) * torch.log(1 - p))
        train_loss = bce[is_train].mean() if is_train.any() else bce.mean()
        loss = train_loss + l2_sigma * (log_sigma**2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sigma = torch.exp(log_sigma)
        scale = sigma.mean()
        mu_c = (mu - mu.mean()) / scale
        sigma_c = sigma / scale
        Phat = predict_pref_matrix(mu_c.cpu().numpy(), sigma_c.cpu().numpy())
        eval_mask = ~is_train if (n_test > 0 and (~is_train).any()) else torch.ones_like(is_train)
        pred = (
            _phi((mu_c[ii] - mu_c[jj]) / torch.sqrt(sigma_c[ii] ** 2 + sigma_c[jj] ** 2))
            > 0.5
        )
        emp = pp > 0.5
        acc = (pred[eval_mask] == emp[eval_mask]).double().mean().item()

    return {
        "mu": mu_c.detach().cpu().numpy(),
        "sigma": sigma_c.detach().cpu().numpy(),
        "test_accuracy": acc,
        "accuracy_is_heldout": n_test > 0,
        "pred_matrix": Phat,
        "comparison_count": m,
    }
