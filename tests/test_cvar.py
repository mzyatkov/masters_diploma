"""Tests for CVaR definition."""

from __future__ import annotations

import numpy as np
import pytest

from src.monte_carlo import compute_distribution_stats, compute_mean_and_cvar


def test_cvar_worst_tail_mean() -> None:
    samples = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    alpha = 0.4
    mean, cvar = compute_mean_and_cvar(samples, alpha=alpha)
    assert mean == 30.0
    # worst ceil(0.4*5)=2 values: 10,20 mean=15
    assert cvar == pytest.approx(15.0)


def test_cvar_alpha_invalid() -> None:
    with pytest.raises(ValueError):
        compute_mean_and_cvar(np.array([1.0]), alpha=0.0)


def test_distribution_stats_contains_ci_fields() -> None:
    samples = np.array([5.0, 8.0, 12.0, 20.0, -1.0, 4.0], dtype=float)
    rng = np.random.default_rng(123)
    stats = compute_distribution_stats(
        samples,
        alpha=0.2,
        ci_level=0.9,
        n_bootstrap=80,
        rng=rng,
    )
    assert stats["mean_ci_low"] <= stats["mean"] <= stats["mean_ci_high"]
    assert stats["cvar_ci_low"] <= stats["cvar"] <= stats["cvar_ci_high"]
    assert stats["p5"] <= stats["p50"] <= stats["p95"]
