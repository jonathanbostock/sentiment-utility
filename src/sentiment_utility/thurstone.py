from __future__ import annotations
import numpy as np
import torch

_NORMAL = torch.distributions.Normal(0.0, 1.0)


def _phi(x: torch.Tensor) -> torch.Tensor:
    return _NORMAL.cdf(x)


def predict_pref_matrix(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    sig_t = torch.as_tensor(sigma, dtype=torch.float64)
    diff = mu_t[:, None] - mu_t[None, :]
    denom = torch.sqrt(sig_t[:, None] ** 2 + sig_t[None, :] ** 2)
    P = _phi(diff / denom)
    P.fill_diagonal_(0.5)
    return P.numpy()


def fit_thurstone(pref: np.ndarray, lr: float = 0.05, steps: int = 2000,
                  test_frac: float = 0.0, l2_sigma: float = 0.01,
                  seed: int = 0) -> dict:
    torch.manual_seed(seed)
    n = pref.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.as_tensor(pref, dtype=torch.float64, device=device)

    # off-diagonal mask, optional train/test split over ordered pairs
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    idx = mask.nonzero(as_tuple=False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(idx.shape[0], generator=g)
    n_test = int(test_frac * idx.shape[0])
    test_idx = idx[perm[:n_test]]
    train_idx = idx[perm[n_test:]]

    train_mask = torch.zeros_like(mask)
    train_mask[train_idx[:, 0], train_idx[:, 1]] = True

    mu = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    log_sigma = torch.zeros(n, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        sigma = torch.exp(log_sigma)
        diff = mu[:, None] - mu[None, :]
        denom = torch.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
        p = _phi(diff / denom).clamp(1e-6, 1 - 1e-6)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        loss = bce[train_mask].mean() + l2_sigma * (log_sigma ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        sigma = torch.exp(log_sigma)
        mu_c = mu - mu.mean()  # center for identifiability
        Phat = predict_pref_matrix(mu_c.cpu().numpy(), sigma.cpu().numpy())
        # test accuracy: thresholded predicted vs empirical on held-out (or all) pairs
        eval_idx = test_idx if n_test > 0 else idx
        ph = torch.as_tensor(Phat, device=device)
        pred_label = (ph[eval_idx[:, 0], eval_idx[:, 1]] > 0.5).double()
        emp_label = (target[eval_idx[:, 0], eval_idx[:, 1]] > 0.5).double()
        acc = (pred_label == emp_label).double().mean().item()

    return {
        "mu": mu_c.detach().cpu().numpy(),
        "sigma": sigma.detach().cpu().numpy(),
        "test_accuracy": acc,
        "pred_matrix": Phat,
    }
