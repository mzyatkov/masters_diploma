"""Economic simulation for maintenance and procurement policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyVector:
    """Interpretable decision variables for one pool of disks."""

    tau_replace: float  # preventive replacement if predicted p > tau
    safety_stock: float  # reorder when spare inventory drops below this
    order_qty: float  # order size when reordering


def _sample_dist(spec: dict[str, Any], rng: np.random.Generator) -> float:
    """Sample one economic parameter from config ``spec``."""
    kind = spec["kind"]
    if kind == "fixed":
        return float(spec["value"])
    if kind == "normal":
        return float(rng.normal(spec["mean"], spec["std"]))
    if kind == "lognormal":
        # parameters of underlying Normal for log(X)
        return float(rng.lognormal(spec["mean"], spec["sigma"]))
    if kind == "triangular":
        return float(rng.triangular(spec["left"], spec["mode"], spec["right"]))
    if kind == "poisson":
        return float(rng.poisson(spec["mu"]))
    raise ValueError(f"Unknown distribution kind: {kind}")


def sample_economic_parameters(
    economics_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Draw one joint sample of economic parameters for a Monte Carlo run.

    Keys follow ``configs/default.yaml`` under ``economics``.
    """
    out: dict[str, float] = {}
    out["revenue_per_disk_per_period"] = float(economics_cfg.get("revenue_per_disk_per_period", 0.0))

    for key in [
        "purchase_cost",
        "replacement_cost",
        "emergency_replacement_cost",
        "downtime_cost",
        "holding_cost_per_unit_per_period",
        "salvage_value",
        "lead_time_periods",
    ]:
        if key in economics_cfg:
            out[key] = _sample_dist(economics_cfg[key], rng)

    return out


def _reliability_modifier(model_type: str, cfg: dict[str, Any]) -> float:
    m = cfg.get("reliability_modifier_by_model", {})
    return float(m.get(str(model_type), 1.0))


def simulate_policy_once(
    failure_probs: np.ndarray,
    model_types: np.ndarray | pd.Series,
    policy: PolicyVector,
    economics: dict[str, float],
    economics_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    """
    Simulate **one period** profit for a fleet of disks under a fixed policy.

    **Profit formula (documented assumptions):**

    For each disk *i* with predicted failure probability ``p_i`` (within horizon):

    1. **Preventive path**: if ``p_i > tau_replace``, schedule preventive replacement.
       - Cost: ``replacement_cost + purchase_cost - salvage_value`` (net OPEX).
       - Revenue: ``revenue_per_disk_per_period`` if disk still considered in service
         after the swap (simplified: we still count revenue for the period).

    2. **No preventive path**: the disk may fail stochastically with probability ``p_i``.
       - If failure: pay ``emergency_replacement_cost + downtime_cost + purchase_cost - salvage_value``.
       - If no failure: pay only baseline operating costs (here: zero extra, revenue kept).

    3. **Inventory**: after processing replacements, consumption reduces spare stock.
       If ``inventory < safety_stock``, order ``order_qty`` units at ``purchase_cost``
       each with lead time ``lead_time_periods`` (affects holding cost proxy only).

    4. **Holding cost**: ``holding_cost_per_unit_per_period * inventory_end``.

    This is a **single-period** teaching model; multi-period dynamics would require
    explicit time indexing. For disks without preventive replacement, failure is drawn
    as Bernoulli(``p_eff``) where ``p_eff`` combines predicted risk with a disk-type
    reliability modifier.

    Parameters
    ----------
    failure_probs
        Per-disk failure probabilities for the horizon (same length as fleet).
    model_types
        Categorical labels per disk (for reliability modifiers).
    """
    n = len(failure_probs)
    if len(model_types) != n:
        raise ValueError("model_types length must match failure_probs")

    tau = float(policy.tau_replace)
    safety = float(policy.safety_stock)
    order_q = int(max(1, round(policy.order_qty)))

    rev = float(economics.get("revenue_per_disk_per_period", 0.0))
    pc = float(economics["purchase_cost"])
    rc = float(economics["replacement_cost"])
    ec = float(economics["emergency_replacement_cost"])
    dc = float(economics["downtime_cost"])
    hv = float(economics["holding_cost_per_unit_per_period"])
    sv = float(economics["salvage_value"])

    total_revenue = 0.0
    total_cost = 0.0
    spares_consumed = 0

    for i in range(n):
        p = float(np.clip(failure_probs[i], 1e-6, 1.0 - 1e-6))
        mod = _reliability_modifier(str(model_types[i]), economics_cfg)
        p_eff = float(np.clip(p * mod, 1e-6, 1.0 - 1e-6))

        fail = bool(rng.random() < p_eff)

        if p > tau:
            # preventive replacement
            total_revenue += rev
            total_cost += rc + pc - sv
            spares_consumed += 1
        else:
            total_revenue += rev
            if fail:
                total_cost += ec + dc + pc - sv
                spares_consumed += 1
            else:
                total_cost += 0.0

    # Inventory: start at safety + buffer (assumption: initial spares = safety + order_qty)
    inventory = safety + order_q
    inventory -= spares_consumed
    if inventory < safety:
        order_cost = order_q * pc
        total_cost += order_cost
        inventory += order_q

    total_cost += hv * max(0.0, inventory)

    profit = total_revenue - total_cost
    return float(profit)


def economic_metrics_fixed_policy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    policy: PolicyVector,
    economics_cfg: dict[str, Any],
    rng: np.random.Generator,
    n_scenarios: int = 50,
    model_types: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Expected profit under fixed policy using point predictions and repeated economic sampling.
    """
    if model_types is None:
        model_types = np.array(["pro"] * len(y_true))
    profits = []
    for _ in range(n_scenarios):
        econ = sample_economic_parameters(economics_cfg, rng)
        p = simulate_policy_once(
            y_pred_proba,
            model_types,
            policy,
            econ,
            economics_cfg,
            rng,
        )
        profits.append(p)
    return {"mean_profit": float(np.mean(profits)), "std_profit": float(np.std(profits))}
