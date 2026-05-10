"""Monte Carlo over economic parameters and model uncertainty for a single policy."""

from __future__ import annotations

from typing import Any, Sequence

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


def compute_distribution_stats(
    samples: np.ndarray,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    n_bootstrap: int = 400,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Compute robust summary stats for profit samples, including CI for mean and CVaR.

    Returns p5/p50/p95, std, standard errors and bootstrap percentile CIs.
    """
    x = np.asarray(samples, dtype=float).ravel()
    if len(x) == 0:
        raise ValueError("samples is empty.")
    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be in (0,1).")

    mean, cvar = compute_mean_and_cvar(x, alpha=alpha)
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    se_mean = float(std / np.sqrt(len(x))) if len(x) > 0 else 0.0
    p5, p50, p95 = np.percentile(x, [5, 50, 95]).tolist()

    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    b_mean = np.empty(n_bootstrap, dtype=float)
    b_cvar = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        m_b, c_b = compute_mean_and_cvar(xb, alpha=alpha)
        b_mean[i] = m_b
        b_cvar[i] = c_b

    tail = (1.0 - ci_level) / 2.0
    q_low = 100.0 * tail
    q_high = 100.0 * (1.0 - tail)
    mean_ci_low, mean_ci_high = np.percentile(b_mean, [q_low, q_high]).tolist()
    cvar_ci_low, cvar_ci_high = np.percentile(b_cvar, [q_low, q_high]).tolist()
    se_cvar = float(np.std(b_cvar, ddof=1)) if len(b_cvar) > 1 else 0.0

    return {
        "n_samples": float(n),
        "mean": float(mean),
        "cvar": float(cvar),
        "std": std,
        "se_mean": se_mean,
        "se_cvar": se_cvar,
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "mean_ci_low": float(mean_ci_low),
        "mean_ci_high": float(mean_ci_high),
        "cvar_ci_low": float(cvar_ci_low),
        "cvar_ci_high": float(cvar_ci_high),
    }


def evaluate_policies_with_uncertainty(
    mean_pred: np.ndarray,
    variance: np.ndarray | None,
    ensemble: np.ndarray | None,
    model_types: np.ndarray,
    policies: Sequence[Any],
    economics_cfg: dict[str, Any],
    rng: np.random.Generator,
    n_samples: int,
    *,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    n_bootstrap: int = 400,
) -> list[dict[str, float]]:
    """Batch re-evaluate policies and return uncertainty-aware objective summaries."""
    from src.economic_model import PolicyVector

    out: list[dict[str, float]] = []
    for i, policy in enumerate(policies):
        if isinstance(policy, PolicyVector):
            pol = policy
        elif isinstance(policy, dict):
            pol = PolicyVector(
                tau_replace=float(policy["tau_replace"]),
                safety_stock=float(policy["safety_stock"]),
                order_qty=float(policy["order_qty"]),
            )
        else:
            arr = np.asarray(policy, dtype=float).ravel()
            if len(arr) != 3:
                raise ValueError("policy vector must have 3 elements.")
            pol = PolicyVector(
                tau_replace=float(arr[0]),
                safety_stock=float(arr[1]),
                order_qty=float(arr[2]),
            )

        samples = run_monte_carlo_for_policy(
            mean_pred,
            variance,
            ensemble,
            model_types,
            pol,
            economics_cfg,
            rng,
            n_samples,
        )
        stats = compute_distribution_stats(
            samples,
            alpha=alpha,
            ci_level=ci_level,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        out.append(
            {
                "policy_idx": int(i),
                "tau_replace": float(pol.tau_replace),
                "safety_stock": float(pol.safety_stock),
                "order_qty": float(pol.order_qty),
                "expected_profit": float(stats["mean"]),
                "cvar_profit": float(stats["cvar"]),
                "expected_profit_ci_low": float(stats["mean_ci_low"]),
                "expected_profit_ci_high": float(stats["mean_ci_high"]),
                "cvar_profit_ci_low": float(stats["cvar_ci_low"]),
                "cvar_profit_ci_high": float(stats["cvar_ci_high"]),
                "std_profit": float(stats["std"]),
                "se_expected_profit": float(stats["se_mean"]),
                "se_cvar_profit": float(stats["se_cvar"]),
                "p5_profit": float(stats["p5"]),
                "p50_profit": float(stats["p50"]),
                "p95_profit": float(stats["p95"]),
                "n_samples": int(stats["n_samples"]),
            }
        )
    return out
