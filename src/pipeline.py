"""End-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import prepare_dataset
from src.evaluation import evaluate_models
from src.features import feature_matrix_for_catboost, fit_encode_model_type
from src.models.baseline_survival import train_baseline_survival
from src.models.catboost_uncertainty import predict_with_uncertainty, train_catboost_with_uncertainty
from src.monte_carlo import compute_mean_and_cvar, run_monte_carlo_for_policy
from src.optimization import optimize_policy_with_pymoo
from src import plots
from src.economic_model import PolicyVector
from src.utils import ensure_dir, get_logger, save_json, set_global_seed, setup_logging

logger = get_logger(__name__)


def run_full_pipeline(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    Execute full pipeline: data -> models -> evaluation -> Monte Carlo -> Pareto optimization.

    Artifacts go to ``outputs/`` and ``reports/`` under project root.
    """
    setup_logging()
    cfg = load_config(config_path)
    set_global_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    raw = cfg.raw
    bundle = prepare_dataset(raw, random_seed=cfg.seed)

    train, val, test, enc = fit_encode_model_type(bundle.train, bundle.val, bundle.test)
    feat_cfg = raw["features"]
    numeric = list(feat_cfg["numeric_columns"])
    categorical = list(feat_cfg.get("categorical_columns", []))

    X_train, cat_feats = feature_matrix_for_catboost(train, numeric, categorical)
    X_val, _ = feature_matrix_for_catboost(val, numeric, categorical)
    X_test, _ = feature_matrix_for_catboost(test, numeric, categorical)

    y_train = train[bundle.target_binary].values
    y_val = val[bundle.target_binary].values
    y_test = test[bundle.target_binary].values

    # Baseline survival on train
    surv_feats = [c for c in X_train.columns if c in train.columns or c.endswith("_encoded")]
    surv_feats = list(X_train.columns)  # encoded model_type included
    baseline = train_baseline_survival(
        train,
        duration_col=bundle.target_survival_duration,
        event_col=bundle.target_survival_event,
        feature_columns=surv_feats,
        horizon_days=float(raw["project"]["horizon_days"]),
        penalizer=float(raw["model"]["baseline"]["penalizer"]),
        l1_ratio=float(raw["model"]["baseline"]["l1_ratio"]),
    )

    cat_model = train_catboost_with_uncertainty(
        X_train,
        y_train,
        X_val,
        y_val,
        cat_feats,
        raw,
        random_seed=cfg.seed,
    )

    out_dir = ensure_dir(cfg.output_dir)
    rep_dir = ensure_dir(cfg.reports_dir)

    metrics = evaluate_models(
        bundle,
        baseline,
        cat_model,
        X_val,
        y_val,
        X_test,
        y_test,
        raw,
        rng,
        rep_dir,
    )

    # Use test predictions for decision layer (fleet = test rows)
    pred = predict_with_uncertainty(cat_model, X_test)
    mean_p = pred["mean"]
    var_p = pred["variance"]
    ens = pred.get("ensemble")
    model_types = test["model_type"].values

    # Reference Monte Carlo for default policy
    pol = raw["evaluation_policy"]
    policy = PolicyVector(
        tau_replace=float(pol["tau_replace"]),
        safety_stock=float(pol["safety_stock"]),
        order_qty=float(pol["order_qty"]),
    )
    mc_n = int(raw["monte_carlo"]["n_samples"])
    samples = run_monte_carlo_for_policy(
        mean_p,
        var_p,
        ens,
        model_types,
        policy,
        raw["economics"],
        rng,
        mc_n,
    )
    mean_profit, cvar = compute_mean_and_cvar(samples, float(raw["monte_carlo"]["cvar_alpha"]))
    save_json(
        out_dir / "monte_carlo_reference_policy.json",
        {"mean_profit": mean_profit, "cvar": cvar, "policy": pol},
    )

    # Multi-objective optimization (can be heavy; uses same test fleet)
    opt_res = optimize_policy_with_pymoo(mean_p, var_p, ens, model_types, raw, rng)

    pareto_obj = np.column_stack([-opt_res.pareto_F[:, 0], -opt_res.pareto_F[:, 1]])
    pareto_df = pd.DataFrame(
        {
            "tau_replace": opt_res.pareto_X[:, 0],
            "safety_stock": opt_res.pareto_X[:, 1],
            "order_qty": opt_res.pareto_X[:, 2],
            "expected_profit": pareto_obj[:, 0],
            "cvar_profit": pareto_obj[:, 1],
        }
    )
    pareto_df.to_csv(rep_dir / "pareto_front.csv", index=False)

    plots.plot_pareto_front(
        pareto_obj,
        rep_dir / "pareto_front.png",
        highlight_indices={
            "risk-neutral": opt_res.risk_neutral_idx,
            "risk-averse (CVaR)": opt_res.risk_averse_idx,
            "knee": opt_res.knee_idx,
        },
    )
    plots.plot_pareto_population_full(
        opt_res.population_F,
        opt_res.population_rank,
        rep_dir / "pareto_population_full.png",
        highlight_indices_pop=opt_res.highlight_indices_pop or None,
    )
    if len(opt_res.population_F):
        _pop_obj = np.column_stack(
            [-opt_res.population_F[:, 0], -opt_res.population_F[:, 1]]
        )
        pd.DataFrame(
            {
                "expected_profit": _pop_obj[:, 0],
                "cvar_profit": _pop_obj[:, 1],
                "nsga2_rank": opt_res.population_rank,
            }
        ).to_csv(rep_dir / "pareto_population_all.csv", index=False)

    summary = (
        "Model comparison (higher ROC-AUC / lower Brier is better for ranking).\n"
        "CatBoost ensemble typically wins on discrimination; check calibration curves.\n"
        "Pareto front trades expected profit against tail CVaR of profit.\n"
    )
    plots.write_summary_text(rep_dir / "summary.txt", summary)

    result: dict[str, Any] = {
        "metrics_table": metrics.to_dict(orient="records"),
        "reference_mc": {"mean_profit": mean_profit, "cvar": cvar},
        "pareto": pareto_df.to_dict(orient="records"),
        "highlights": {
            "risk_neutral": opt_res.pareto_X[opt_res.risk_neutral_idx].tolist(),
            "risk_averse": opt_res.pareto_X[opt_res.risk_averse_idx].tolist(),
            "knee": opt_res.pareto_X[opt_res.knee_idx].tolist(),
        },
    }
    save_json(out_dir / "pipeline_result.json", result)
    logger.info("Pipeline finished. Artifacts in %s and %s", out_dir, rep_dir)
    return result
