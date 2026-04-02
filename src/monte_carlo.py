"""Monte Carlo over economic parameters and model uncertainty for a single policy."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_mean_and_cvar(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """
    Expected profit and CVaR_alpha (lower tail of profit distribution).

    **Definition (CVaR for profit):**

    - Sort samples ascending (worst profits first).
    - Take the worst ``ceil(alpha * N)`` samples (or ``max(1, ...)``).
    - CVaR is their arithmetic mean.

    For negative alpha or invalid input, raises ``ValueError``.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    if n == 0:
        raise ValueError("samples is empty.")
    sorted_x = np.sort(x)
    k = max(1, int(np.ceil(alpha * n)))
    worst = sorted_x[:k]
    cvar = float(worst.mean())
    mean = float(sorted_x.mean())
    return mean, cvar


def sample_failure_probs_from_predictive(
    mean_p: np.ndarray,
    variance: np.ndarray | None,
    ensemble: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one vector of per-disk failure probabilities.

    Prefer ``ensemble`` rows (sample a member index per disk); else Beta approximation
    from mean/variance (clipped to (0,1)).
    """
    n = len(mean_p)
    if ensemble is not None and ensemble.shape[0] == n:
        k = ensemble.shape[1]
        idx = rng.integers(0, k, size=n)
        return np.clip(ensemble[np.arange(n), idx], 1e-6, 1 - 1e-6)

    m = np.clip(mean_p, 1e-6, 1 - 1e-6)
    if variance is None:
        return m
    v = np.clip(variance, 1e-6, m * (1 - m))
    # Method of moments for Beta
    common = m * (1 - m) / v - 1
    common = np.maximum(common, 1e-3)
    a = m * common
    b = (1 - m) * common
    return rng.beta(a, b)


def run_monte_carlo_for_policy(
    mean_pred: np.ndarray,
    variance: np.ndarray | None,
    ensemble: np.ndarray | None,
    model_types: np.ndarray,
    policy: Any,
    economics_cfg: dict[str, Any],
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """
    Run ``n_samples`` economic simulations for the given policy.

    Each iteration: sample economic parameters, sample per-disk failure probabilities
    from predictive (ensemble or Beta), then ``simulate_policy_once``.
    """
    from src.economic_model import PolicyVector, sample_economic_parameters, simulate_policy_once

    if not isinstance(policy, PolicyVector):
        policy = PolicyVector(
            tau_replace=float(policy["tau_replace"]),
            safety_stock=float(policy["safety_stock"]),
            order_qty=float(policy["order_qty"]),
        )

    profits = np.empty(n_samples, dtype=float)
    for t in range(n_samples):
        econ = sample_economic_parameters(economics_cfg, rng)
        p_vec = sample_failure_probs_from_predictive(mean_pred, variance, ensemble, rng)
        profits[t] = simulate_policy_once(
            p_vec,
            model_types,
            policy,
            econ,
            economics_cfg,
            rng,
        )
    return profits
