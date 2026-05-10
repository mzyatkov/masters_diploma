"""Lightweight integration smoke tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.config import load_yaml
from src.data import prepare_dataset
from src.features import feature_matrix_for_catboost, fit_encode_model_type
from src.models.catboost_uncertainty import predict_with_uncertainty, train_catboost_with_uncertainty
from src.monte_carlo import (
    compute_mean_and_cvar,
    evaluate_policies_with_uncertainty,
    run_monte_carlo_for_policy,
)
from src.optimization import optimize_policy_with_pymoo
from src.utils import project_root


@pytest.fixture
def tiny_config() -> dict:
    path = project_root() / "configs" / "default.yaml"
    raw = load_yaml(path)
    raw = copy.deepcopy(raw)
    raw["data"]["synthetic"]["n_disks"] = 80
    raw["data"]["synthetic"]["snapshots_per_disk_max"] = 4
    raw["model"]["catboost"]["iterations"] = 50
    raw["model"]["catboost"]["bootstrap_ensemble_size"] = 2
    raw["monte_carlo"]["n_samples"] = 30
    raw["monte_carlo"]["n_samples_optimization"] = 20
    raw["optimization"]["population_size"] = 8
    raw["optimization"]["n_generations"] = 2
    raw["optimization"]["robust"]["enabled"] = True
    raw["reliability"]["bootstrap_iterations"] = 30
    raw["reliability"]["pareto_reeval_samples"] = 20
    return raw


def test_prepare_and_models(tiny_config: dict) -> None:
    seed = int(tiny_config["project"]["random_seed"])
    bundle = prepare_dataset(tiny_config, random_seed=seed)
    train, val, test, _ = fit_encode_model_type(bundle.train, bundle.val, bundle.test)
    feat_cfg = tiny_config["features"]
    numeric = list(feat_cfg["numeric_columns"])
    categorical = list(feat_cfg.get("categorical_columns", []))
    X_train, cat_feats = feature_matrix_for_catboost(train, numeric, categorical)
    X_val, _ = feature_matrix_for_catboost(val, numeric, categorical)
    X_test, _ = feature_matrix_for_catboost(test, numeric, categorical)
    y_train = train[bundle.target_binary].values
    y_val = val[bundle.target_binary].values

    cat = train_catboost_with_uncertainty(
        X_train,
        y_train,
        X_val,
        y_val,
        cat_feats,
        tiny_config,
        random_seed=seed,
    )
    pred = predict_with_uncertainty(cat, X_test)
    assert pred["mean"].shape[0] == len(X_test)

    rng = np.random.default_rng(seed)
    from src.economic_model import PolicyVector

    pol = tiny_config["evaluation_policy"]
    policy = PolicyVector(
        tau_replace=float(pol["tau_replace"]),
        safety_stock=float(pol["safety_stock"]),
        order_qty=float(pol["order_qty"]),
    )
    samples = run_monte_carlo_for_policy(
        pred["mean"],
        pred["variance"],
        pred.get("ensemble"),
        test["model_type"].values,
        policy,
        tiny_config["economics"],
        rng,
        int(tiny_config["monte_carlo"]["n_samples"]),
    )
    m, c = compute_mean_and_cvar(samples, float(tiny_config["monte_carlo"]["cvar_alpha"]))
    assert np.isfinite(m) and np.isfinite(c)

    # Batch uncertainty-aware policy re-evaluation
    policies = [
        {"tau_replace": 0.35, "safety_stock": 3.0, "order_qty": 15.0},
        {"tau_replace": 0.55, "safety_stock": 6.0, "order_qty": 25.0},
    ]
    reevaluated = evaluate_policies_with_uncertainty(
        pred["mean"],
        pred["variance"],
        pred.get("ensemble"),
        test["model_type"].values,
        policies,
        tiny_config["economics"],
        rng,
        n_samples=20,
        alpha=float(tiny_config["monte_carlo"]["cvar_alpha"]),
        ci_level=float(tiny_config["reliability"]["ci_level"]),
        n_bootstrap=int(tiny_config["reliability"]["bootstrap_iterations"]),
    )
    assert len(reevaluated) == 2
    assert "expected_profit_ci_low" in reevaluated[0]

    # Robust optimization should run in smoke setup
    opt = optimize_policy_with_pymoo(
        pred["mean"],
        pred["variance"],
        pred.get("ensemble"),
        test["model_type"].values,
        tiny_config,
        rng,
    )
    assert len(opt.pareto_X) > 0
