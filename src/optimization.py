"""Multi-objective policy optimization with pymoo (NSGA-II)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.economic_model import PolicyVector
from src.monte_carlo import compute_mean_and_cvar, run_monte_carlo_for_policy
from src.utils import get_logger

logger = get_logger(__name__)


def _mc_seed_for_policy(x_row: np.ndarray, base_seed: int) -> int:
    """Stable integer seed from decision vector (deterministic objectives for same x)."""
    rounded = np.round(np.asarray(x_row, dtype=np.float64), 6).tobytes()
    h = int.from_bytes(hashlib.md5(rounded).digest()[:6], "little")
    return int((base_seed * 1_000_003 + h) % (2**31 - 1))


@dataclass
class OptimizationResult:
    """Pareto set and highlighted solutions."""

    pareto_X: np.ndarray
    pareto_F: np.ndarray  # columns: [-E[profit], -CVaR] (minimization)
    risk_neutral_idx: int
    risk_averse_idx: int
    knee_idx: int
    # Final NSGA-II generation (for visualization of dominated vs non-dominated)
    population_F: np.ndarray  # shape (N, 2), minimization space
    population_rank: np.ndarray  # shape (N,), NSGA-II rank (0 = first front)
    population_X: np.ndarray  # shape (N, 3), decision vectors
    highlight_indices_pop: dict[str, int]  # label -> row index in population (for scatter)


def _pop_row_for_pareto_row(pareto_x_row: np.ndarray, pop_X: np.ndarray) -> int:
    """Index of the nearest row in final population (exact match after optimization)."""
    return int(np.argmin(np.linalg.norm(pop_X - pareto_x_row[None, :], axis=1)))


def _knee_index(F: np.ndarray) -> int:
    """Simple knee: minimize distance to ideal point (min f1, min f2) in objective space."""
    # F is minimization objectives (both positive costs of neg profit)
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    rng = np.maximum(nadir - ideal, 1e-9)
    normalized = (F - ideal) / rng
    dist = np.linalg.norm(normalized, axis=1)
    return int(np.argmin(dist))


class PolicyProblem(Problem):
    """Minimize (-E[profit], -CVaR) via Monte Carlo for each candidate policy."""

    def __init__(
        self,
        mean_pred: np.ndarray,
        variance: np.ndarray | None,
        ensemble: np.ndarray | None,
        model_types: np.ndarray,
        economics_cfg: dict[str, Any],
        rng: np.random.Generator,
        mc_samples: int,
        cvar_alpha: float,
        budget_constraint: float | None,
        *,
        deterministic_mc_per_policy: bool,
        mc_base_seed: int,
        robust_enabled: bool,
        robust_k_mean: float,
        robust_k_cvar: float,
    ) -> None:
        self._mean = mean_pred
        self._var = variance
        self._ens = ensemble
        self._model_types = model_types
        self._econ = economics_cfg
        self._rng = rng
        self._mc_samples = mc_samples
        self._alpha = cvar_alpha
        self._budget = budget_constraint
        self._det_mc = deterministic_mc_per_policy
        self._mc_base_seed = int(mc_base_seed)
        self._robust_enabled = bool(robust_enabled)
        self._robust_k_mean = float(robust_k_mean)
        self._robust_k_cvar = float(robust_k_cvar)

        super().__init__(
            n_var=3,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.array([0.0, 0.0, 1.0]),
            xu=np.array([1.0, 50.0, 80.0]),
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        n = x.shape[0]
        f1 = np.empty(n)
        f2 = np.empty(n)
        for i in range(n):
            pol = PolicyVector(
                tau_replace=float(x[i, 0]),
                safety_stock=float(x[i, 1]),
                order_qty=float(x[i, 2]),
            )
            mc_rng = (
                np.random.default_rng(_mc_seed_for_policy(x[i], self._mc_base_seed))
                if self._det_mc
                else self._rng
            )
            samples = run_monte_carlo_for_policy(
                self._mean,
                self._var,
                self._ens,
                self._model_types,
                pol,
                self._econ,
                mc_rng,
                self._mc_samples,
            )
            mean_profit, cvar = compute_mean_and_cvar(samples, self._alpha)
            if self._robust_enabled:
                std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
                se_mean = std / np.sqrt(max(1, len(samples)))
                k_tail = max(1, int(np.ceil(self._alpha * len(samples))))
                worst = np.sort(samples)[:k_tail]
                se_cvar = (
                    float(np.std(worst, ddof=1)) / np.sqrt(max(1, len(worst)))
                    if len(worst) > 1
                    else 0.0
                )
                robust_mean = mean_profit - self._robust_k_mean * se_mean
                robust_cvar = cvar - self._robust_k_cvar * se_cvar
                f1[i] = -robust_mean
                f2[i] = -robust_cvar
            else:
                f1[i] = -mean_profit
                f2[i] = -cvar
            if self._budget is not None:
                # Soft penalty: crude proxy spend = tau-related replacements + orders
                spend = float(x[i, 0]) * 5000 + float(x[i, 2]) * 50
                if spend > self._budget:
                    f1[i] += 1e6
                    f2[i] += 1e6
        out["F"] = np.column_stack([f1, f2])


def optimize_policy_with_pymoo(
    mean_pred: np.ndarray,
    variance: np.ndarray | None,
    ensemble: np.ndarray | None,
    model_types: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> OptimizationResult:
    """
    NSGA-II on (tau, safety_stock, order_qty) to maximize mean profit and CVaR.

    Objectives passed to pymoo are ``-E[profit]`` and ``-CVaR`` (minimization).

    When ``optimization.deterministic_mc_per_policy`` is true (default), each
    candidate uses a fixed RNG derived from its decision vector so dominance is
    not polluted by Monte Carlo noise between evaluations.
    """
    mc_cfg = config["monte_carlo"]
    opt_cfg = config["optimization"]
    n_samples = int(mc_cfg.get("n_samples_optimization", mc_cfg["n_samples"]))
    alpha = float(mc_cfg["cvar_alpha"])
    budget = opt_cfg.get("budget_constraint")
    budget_f = float(budget) if budget is not None else None

    det_mc = bool(opt_cfg.get("deterministic_mc_per_policy", True))
    mc_base = int(opt_cfg.get("seed", 42))
    robust_cfg = opt_cfg.get("robust", {})
    robust_enabled = bool(robust_cfg.get("enabled", False))
    robust_k_mean = float(robust_cfg.get("k_mean", 0.0))
    robust_k_cvar = float(robust_cfg.get("k_cvar", 0.0))
    logger.info(
        "Optimization robust mode: enabled=%s, k_mean=%.3f, k_cvar=%.3f",
        robust_enabled,
        robust_k_mean,
        robust_k_cvar,
    )

    problem = PolicyProblem(
        mean_pred,
        variance,
        ensemble,
        model_types,
        config["economics"],
        rng,
        n_samples,
        alpha,
        budget_f,
        deterministic_mc_per_policy=det_mc,
        mc_base_seed=mc_base,
        robust_enabled=robust_enabled,
        robust_k_mean=robust_k_mean,
        robust_k_cvar=robust_k_cvar,
    )

    algorithm = NSGA2(pop_size=int(opt_cfg["population_size"]))
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", int(opt_cfg["n_generations"])),
        seed=int(opt_cfg.get("seed", 42)),
        verbose=False,
    )

    X = res.X
    F = res.F
    if X is None or F is None or len(X) == 0:
        raise RuntimeError("Optimization failed to produce a Pareto set.")

    # Risk-neutral: best E[profit] => minimize f1
    risk_neutral_idx = int(np.argmin(F[:, 0]))
    # Risk-averse: best CVaR => minimize f2 (most negative tail)
    risk_averse_idx = int(np.argmin(F[:, 1]))
    knee_idx = _knee_index(F)

    logger.info(
        "Pareto size=%d, risk-neutral idx=%d, risk-averse idx=%d, knee idx=%d",
        len(X),
        risk_neutral_idx,
        risk_averse_idx,
        knee_idx,
    )

    _pop = getattr(res, "pop", None)
    if _pop is None or len(_pop) == 0:
        pop_F = np.empty((0, 2), dtype=float)
        pop_rank = np.empty(0, dtype=int)
        pop_X = np.empty((0, 3), dtype=float)
    else:
        pop_F = np.asarray(_pop.get("F"), dtype=float)
        pop_rank = np.asarray(_pop.get("rank"), dtype=int).ravel()
        pop_X = np.asarray(_pop.get("X"), dtype=float)

    if len(pop_X) == 0:
        highlight_pop: dict[str, int] = {}
    else:
        highlight_pop = {
            "risk-neutral": _pop_row_for_pareto_row(X[risk_neutral_idx], pop_X),
            "risk-averse (CVaR)": _pop_row_for_pareto_row(X[risk_averse_idx], pop_X),
            "knee": _pop_row_for_pareto_row(X[knee_idx], pop_X),
        }

    return OptimizationResult(
        pareto_X=X,
        pareto_F=F,
        risk_neutral_idx=risk_neutral_idx,
        risk_averse_idx=risk_averse_idx,
        knee_idx=knee_idx,
        population_F=pop_F,
        population_rank=pop_rank,
        population_X=pop_X,
        highlight_indices_pop=highlight_pop,
    )
