#!/usr/bin/env python3
"""Run Pareto optimization on test-set predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import prepare_dataset
from src.features import feature_matrix_for_catboost, fit_encode_model_type
from src.models.catboost_uncertainty import predict_with_uncertainty, train_catboost_with_uncertainty
from src.optimization import optimize_policy_with_pymoo
from src import plots
from src.utils import ensure_dir, set_global_seed, setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    pcfg = load_config(args.config)
    set_global_seed(pcfg.seed)
    rng = np.random.default_rng(pcfg.seed)
    raw = pcfg.raw
    bundle = prepare_dataset(raw, random_seed=pcfg.seed)

    train, val, test, _ = fit_encode_model_type(bundle.train, bundle.val, bundle.test)
    feat_cfg = raw["features"]
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
        raw,
        random_seed=pcfg.seed,
    )
    pred = predict_with_uncertainty(cat, X_test)
    model_types = test["model_type"].values

    res = optimize_policy_with_pymoo(
        pred["mean"],
        pred["variance"],
        pred.get("ensemble"),
        model_types,
        raw,
        rng,
    )

    pareto_obj = np.column_stack([-res.pareto_F[:, 0], -res.pareto_F[:, 1]])
    df = pd.DataFrame(
        {
            "tau_replace": res.pareto_X[:, 0],
            "safety_stock": res.pareto_X[:, 1],
            "order_qty": res.pareto_X[:, 2],
            "expected_profit": pareto_obj[:, 0],
            "cvar_profit": pareto_obj[:, 1],
        }
    )
    rep = ensure_dir(pcfg.reports_dir)
    df.to_csv(rep / "pareto_front.csv", index=False)
    plots.plot_pareto_front(
        pareto_obj,
        rep / "pareto_front.png",
        highlight_indices={
            "risk-neutral": res.risk_neutral_idx,
            "risk-averse (CVaR)": res.risk_averse_idx,
            "knee": res.knee_idx,
        },
    )
    plots.plot_pareto_population_full(
        res.population_F,
        res.population_rank,
        rep / "pareto_population_full.png",
        highlight_indices_pop=res.highlight_indices_pop or None,
    )
    if len(res.population_F):
        _po = np.column_stack([-res.population_F[:, 0], -res.population_F[:, 1]])
        pd.DataFrame(
            {
                "expected_profit": _po[:, 0],
                "cvar_profit": _po[:, 1],
                "nsga2_rank": res.population_rank,
            }
        ).to_csv(rep / "pareto_population_all.csv", index=False)
    with open(rep / "optimization_highlights.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "risk_neutral": res.pareto_X[res.risk_neutral_idx].tolist(),
                "risk_averse": res.pareto_X[res.risk_averse_idx].tolist(),
                "knee": res.pareto_X[res.knee_idx].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Pareto saved to {rep}")


if __name__ == "__main__":
    main()
