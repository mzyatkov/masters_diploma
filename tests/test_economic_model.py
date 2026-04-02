"""Economic simulation smoke tests."""

from __future__ import annotations

import numpy as np

from src.economic_model import PolicyVector, sample_economic_parameters, simulate_policy_once


def test_simulate_policy_once_runs() -> None:
    rng = np.random.default_rng(0)
    economics_cfg = {
        "revenue_per_disk_per_period": 100.0,
        "purchase_cost": {"kind": "fixed", "value": 50.0},
        "replacement_cost": {"kind": "fixed", "value": 10.0},
        "emergency_replacement_cost": {"kind": "fixed", "value": 30.0},
        "downtime_cost": {"kind": "fixed", "value": 20.0},
        "holding_cost_per_unit_per_period": {"kind": "fixed", "value": 1.0},
        "salvage_value": {"kind": "fixed", "value": 5.0},
        "lead_time_periods": {"kind": "fixed", "value": 0.0},
        "reliability_modifier_by_model": {"pro": 1.0},
    }
    econ = sample_economic_parameters(economics_cfg, rng)
    p = simulate_policy_once(
        np.array([0.1, 0.9]),
        np.array(["pro", "pro"]),
        PolicyVector(tau_replace=0.5, safety_stock=1.0, order_qty=2.0),
        econ,
        economics_cfg,
        rng,
    )
    assert isinstance(p, float)
