"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import load_config
from src.data import prepare_dataset
from src.evaluation import evaluate_models
from src.features import feature_matrix_for_catboost, fit_encode_model_type
from src.models.baseline_survival import train_baseline_survival
from src.models.catboost_uncertainty import predict_with_uncertainty, train_catboost_with_uncertainty
from src.monte_carlo import (
    compute_mean_and_cvar,
    evaluate_policies_with_uncertainty,
    run_monte_carlo_for_policy,
)
from src.optimization import optimize_policy_with_pymoo
from src import plots
from src.economic_model import PolicyVector
from src.utils import ensure_dir, get_logger, save_json, set_global_seed, setup_logging

logger = get_logger(__name__)


def _append_debug_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": "b4163a",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open("/Users/cobeq/Documents/diploma_v2/.cursor/debug-b4163a.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _nearest_row_idx(candidates: np.ndarray, point: np.ndarray) -> int:
    """Index of closest candidate vector in Euclidean norm."""
    if len(candidates) == 0:
        return 0
    d = np.linalg.norm(candidates - point[None, :], axis=1)
    return int(np.argmin(d))


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

    # Calibrate predictive probabilities for the decision layer using validation set.
    # This aligns policy optimization with the best probabilistic estimate.
    if y_val is not None and len(np.unique(y_val)) >= 2:
        p_val_raw = predict_with_uncertainty(cat_model, X_val)["mean"]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.clip(p_val_raw, 1e-6, 1 - 1e-6), y_val)

        mean_p = np.clip(iso.predict(np.clip(mean_p, 1e-6, 1 - 1e-6)), 1e-6, 1 - 1e-6)

        if ens is not None and ens.ndim == 2:
            ens_cal = np.empty_like(ens)
            for j in range(ens.shape[1]):
                ens_cal[:, j] = np.clip(
                    iso.predict(np.clip(ens[:, j], 1e-6, 1 - 1e-6)),
                    1e-6,
                    1 - 1e-6,
                )
            ens = ens_cal
            var_p = ens.var(axis=1, ddof=0)

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
    # region agent log
    _append_debug_log(
        hypothesis_id="H1",
        location="src/pipeline.py:opt_result",
        message="Optimization outputs for plotting contexts",
        data={
            "pareto_size": int(len(opt_res.pareto_X)),
            "population_size": int(len(opt_res.population_X)),
            "rank0_count": int(np.sum(np.asarray(opt_res.population_rank) == 0)),
            "robust_enabled": bool(raw.get("optimization", {}).get("robust", {}).get("enabled", False)),
        },
        run_id="post-fix",
    )
    # endregion

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

    reliability_cfg = raw.get("reliability", {})
    reevaluate_samples = int(
        reliability_cfg.get("pareto_reeval_samples", raw["monte_carlo"]["n_samples"])
    )
    ci_level = float(reliability_cfg.get("ci_level", 0.95))
    n_bootstrap = int(reliability_cfg.get("bootstrap_iterations", 300))
    pareto_policies = [
        {
            "tau_replace": float(x[0]),
            "safety_stock": float(x[1]),
            "order_qty": float(x[2]),
        }
        for x in opt_res.pareto_X
    ]
    reevaluated = evaluate_policies_with_uncertainty(
        mean_p,
        var_p,
        ens,
        model_types,
        pareto_policies,
        raw["economics"],
        rng,
        reevaluate_samples,
        alpha=float(raw["monte_carlo"]["cvar_alpha"]),
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )
    pareto_unc_df = pd.DataFrame(reevaluated)
    if not pareto_unc_df.empty:
        pareto_unc_df["robust_score"] = (
            pareto_unc_df["expected_profit_ci_low"] + pareto_unc_df["cvar_profit_ci_low"]
        )
        robust_idx = int(pareto_unc_df["robust_score"].idxmax())
    else:
        robust_idx = 0
    pareto_unc_df.to_csv(rep_dir / "pareto_front_with_uncertainty.csv", index=False)

    # For visualization, broaden beyond strict rank-0 front when it collapses to 1 point.
    max_rank = int(reliability_cfg.get("plot_candidates_max_rank", 2))
    top_k = int(reliability_cfg.get("plot_candidates_top_k", 25))
    pop_F = np.asarray(opt_res.population_F, dtype=float)
    pop_rank = np.asarray(opt_res.population_rank, dtype=int)
    pop_X = np.asarray(opt_res.population_X, dtype=float)
    mask = pop_rank <= max_rank
    if np.any(mask):
        cand_X = pop_X[mask]
        cand_F = pop_F[mask]
        cand_rank = pop_rank[mask]
    else:
        cand_X = pop_X
        cand_F = pop_F
        cand_rank = pop_rank
    if len(cand_X):
        order = np.lexsort(((cand_F[:, 0] + cand_F[:, 1]), cand_rank))
        keep = order[: min(top_k, len(order))]
        viz_X = cand_X[keep]
    else:
        viz_X = opt_res.pareto_X
    # region agent log
    _append_debug_log(
        hypothesis_id="H1",
        location="src/pipeline.py:viz_selection",
        message="Candidate subset for errorbar figure",
        data={
            "max_rank": int(max_rank),
            "top_k": int(top_k),
            "cand_size": int(len(cand_X)),
            "viz_size": int(len(viz_X)),
            "using_all_pareto_fallback": bool(len(cand_X) == 0),
        },
        run_id="post-fix",
    )
    # endregion

    viz_policies = [
        {"tau_replace": float(x[0]), "safety_stock": float(x[1]), "order_qty": float(x[2])}
        for x in viz_X
    ]
    viz_eval = evaluate_policies_with_uncertainty(
        mean_p,
        var_p,
        ens,
        model_types,
        viz_policies,
        raw["economics"],
        rng,
        reevaluate_samples,
        alpha=float(raw["monte_carlo"]["cvar_alpha"]),
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )
    viz_unc_df = pd.DataFrame(viz_eval)
    if not viz_unc_df.empty:
        viz_unc_df["robust_score"] = (
            viz_unc_df["expected_profit_ci_low"] + viz_unc_df["cvar_profit_ci_low"]
        )
        viz_robust_idx = int(viz_unc_df["robust_score"].idxmax())
    else:
        viz_robust_idx = 0
    viz_unc_df.to_csv(rep_dir / "pareto_candidates_with_uncertainty.csv", index=False)
    # region agent log
    _append_debug_log(
        hypothesis_id="H4",
        location="src/pipeline.py:reeval_delta",
        message="Delta between optimization front and reevaluated candidates",
        data={
            "pareto_front_first": (
                {
                    "expected_profit": float(pareto_df.iloc[0]["expected_profit"]),
                    "cvar_profit": float(pareto_df.iloc[0]["cvar_profit"]),
                }
                if len(pareto_df)
                else None
            ),
            "viz_first": (
                {
                    "expected_profit": float(viz_unc_df.iloc[0]["expected_profit"]),
                    "cvar_profit": float(viz_unc_df.iloc[0]["cvar_profit"]),
                }
                if len(viz_unc_df)
                else None
            ),
        },
        run_id="post-fix",
    )
    # endregion

    plots.plot_pareto_front(
        pareto_obj,
        rep_dir / "pareto_front.png",
        highlight_indices={
            "risk-neutral": opt_res.risk_neutral_idx,
            "risk-averse (CVaR)": opt_res.risk_averse_idx,
            "knee": opt_res.knee_idx,
        },
    )
    if not viz_unc_df.empty:
        highlight_idx_viz = {
            "risk-neutral": _nearest_row_idx(viz_X, opt_res.pareto_X[opt_res.risk_neutral_idx]),
            "risk-averse (CVaR)": _nearest_row_idx(viz_X, opt_res.pareto_X[opt_res.risk_averse_idx]),
            "knee": _nearest_row_idx(viz_X, opt_res.pareto_X[opt_res.knee_idx]),
            "robust": viz_robust_idx,
        }
        # region agent log
        _append_debug_log(
            hypothesis_id="H2",
            location="src/pipeline.py:highlight_idx_viz",
            message="Star index mapping in errorbar figure",
            data={
                "run_mode": "rank_lte_2_candidates",
                "errorbar_points": int(len(viz_unc_df)),
                "highlight_idx_viz": {k: int(v) for k, v in highlight_idx_viz.items()},
                "viz_points": [
                    {
                        "idx": int(i),
                        "expected_profit": float(viz_unc_df.iloc[i]["expected_profit"]),
                        "cvar_profit": float(viz_unc_df.iloc[i]["cvar_profit"]),
                    }
                    for i in sorted(
                        set(int(v) for v in highlight_idx_viz.values() if 0 <= int(v) < len(viz_unc_df))
                    )
                ],
            },
            run_id="post-fix",
        )
        # endregion
        plots.plot_pareto_front_with_errorbars(
            viz_unc_df,
            rep_dir / "pareto_front_errorbars.png",
            highlight_indices=highlight_idx_viz,
            title=f"Near-Pareto candidates (rank<={max_rank}) with uncertainty intervals",
        )
    plots.plot_pareto_population_full(
        opt_res.population_F,
        opt_res.population_rank,
        rep_dir / "pareto_population_full.png",
        highlight_indices_pop=opt_res.highlight_indices_pop or None,
    )
    # region agent log
    _append_debug_log(
        hypothesis_id="H5",
        location="src/pipeline.py:population_highlights",
        message="Star rank status in full population figure",
        data={
            "highlight_indices_pop": {k: int(v) for k, v in (opt_res.highlight_indices_pop or {}).items()},
            "highlight_ranks": {
                k: int(opt_res.population_rank[int(v)])
                for k, v in (opt_res.highlight_indices_pop or {}).items()
                if 0 <= int(v) < len(opt_res.population_rank)
            },
        },
        run_id="post-fix",
    )
    # endregion
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

    save_json(
        out_dir / "optimization_highlights_robust.json",
        {
            "risk_neutral": opt_res.pareto_X[opt_res.risk_neutral_idx].tolist(),
            "risk_averse": opt_res.pareto_X[opt_res.risk_averse_idx].tolist(),
            "knee": opt_res.pareto_X[opt_res.knee_idx].tolist(),
            "robust_idx": robust_idx,
            "robust_policy": opt_res.pareto_X[robust_idx].tolist(),
        },
    )

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
        "pareto_with_uncertainty": pareto_unc_df.to_dict(orient="records"),
        "pareto_candidates_with_uncertainty": viz_unc_df.to_dict(orient="records"),
        "highlights": {
            "risk_neutral": opt_res.pareto_X[opt_res.risk_neutral_idx].tolist(),
            "risk_averse": opt_res.pareto_X[opt_res.risk_averse_idx].tolist(),
            "knee": opt_res.pareto_X[opt_res.knee_idx].tolist(),
            "robust": opt_res.pareto_X[robust_idx].tolist(),
        },
    }
    save_json(out_dir / "pipeline_result.json", result)
    logger.info("Pipeline finished. Artifacts in %s and %s", out_dir, rep_dir)
    return result
