"""Model evaluation: ranking metrics, calibration, economic summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data import DatasetBundle
from src.economic_model import (
    PolicyVector,
    economic_metrics_fixed_policy,
    expected_economic_parameters,
    realized_policy_profit_from_outcomes,
)
from src.models.baseline_survival import BaselineSurvivalArtifacts, predict_failure_prob_in_horizon
from src.models.catboost_uncertainty import CatBoostUncertaintyArtifacts, predict_with_uncertainty
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def _ece_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """Expected / max calibration error (simple binning)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    mce = float(np.max(np.abs(prob_true - prob_pred)))
    return ece, mce


def _point_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute all probabilistic classification metrics used in reports."""
    auc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece, mce = _ece_mce(y_true, y_prob)
    return {
        "roc_auc": float(auc),
        "pr_auc": float(pr),
        "brier": float(brier),
        "ece": float(ece),
        "mce": float(mce),
    }


def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Bootstrap CIs for ROC-AUC / PR-AUC / Brier / ECE."""
    n = len(y_true)
    if n == 0:
        raise ValueError("y_true is empty.")
    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be in (0,1).")

    point = _point_metrics(y_true, y_prob)
    keys = list(point.keys())
    arr = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = _point_metrics(yb, pb)
        for k in keys:
            arr[k].append(m[k])

    if len(arr["roc_auc"]) == 0:
        return {
            **{f"{k}_value": float(v) for k, v in point.items()},
            **{f"{k}_ci_low": float(v) for k, v in point.items()},
            **{f"{k}_ci_high": float(v) for k, v in point.items()},
        }

    tail = (1.0 - ci_level) / 2.0
    q_low = 100.0 * tail
    q_high = 100.0 * (1.0 - tail)
    out: dict[str, float] = {}
    for k, v in point.items():
        vals = np.asarray(arr[k], dtype=float)
        lo, hi = np.percentile(vals, [q_low, q_high]).tolist()
        out[f"{k}_value"] = float(v)
        out[f"{k}_ci_low"] = float(lo)
        out[f"{k}_ci_high"] = float(hi)
    return out


def _uncertainty_decomposition(ensemble: np.ndarray) -> pd.DataFrame:
    """
    Decompose predictive uncertainty:
    Var(Y) = E[p(1-p)] + Var(p).
    """
    ens = np.asarray(ensemble, dtype=float)
    if ens.ndim != 2:
        raise ValueError("ensemble must be 2D with shape (n_samples, n_members)")
    p_mean = np.clip(ens.mean(axis=1), 1e-6, 1 - 1e-6)
    epistemic = ens.var(axis=1, ddof=0)
    aleatoric = (ens * (1.0 - ens)).mean(axis=1)
    total = aleatoric + epistemic
    return pd.DataFrame(
        {
            "mean_pred": p_mean,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_uncertainty": total,
        }
    )


def evaluate_models(
    bundle: DatasetBundle,
    baseline: BaselineSurvivalArtifacts,
    catboost: CatBoostUncertaintyArtifacts,
    X_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
    reports_dir: Path,
) -> pd.DataFrame:
    """
    Compare baseline vs CatBoost on held-out test set.

    Saves ROC/PR/calibration plots via ``plots`` module; returns a metrics table.
    """
    from src import plots

    p_base = predict_failure_prob_in_horizon(baseline, X_test)
    pred_cb = predict_with_uncertainty(catboost, X_test)
    p_cat = pred_cb["mean"]

    def _fit_iso(y_true: np.ndarray, p_raw: np.ndarray) -> IsotonicRegression | None:
        if y_true is None or len(np.unique(y_true)) < 2:
            return None
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(np.clip(p_raw, 1e-6, 1 - 1e-6), y_true)
        return model

    calibrated_preds: list[tuple[str, np.ndarray]] = []
    if X_val is not None and y_val is not None:
        p_base_val = predict_failure_prob_in_horizon(baseline, X_val)
        p_cat_val = predict_with_uncertainty(catboost, X_val)["mean"]

        iso_base = _fit_iso(y_val, p_base_val)
        iso_cat = _fit_iso(y_val, p_cat_val)
        if iso_base is not None:
            p_base_iso = np.clip(iso_base.predict(np.clip(p_base, 1e-6, 1 - 1e-6)), 1e-6, 1 - 1e-6)
            calibrated_preds.append(("weibull_aft_isotonic", p_base_iso))
        if iso_cat is not None:
            p_cat_iso = np.clip(iso_cat.predict(np.clip(p_cat, 1e-6, 1 - 1e-6)), 1e-6, 1 - 1e-6)
            calibrated_preds.append(("catboost_ensemble_isotonic", p_cat_iso))

    rows = []
    model_preds = [("weibull_aft", p_base), ("catboost_ensemble", p_cat)] + calibrated_preds
    for name, p in model_preds:
        pm = _point_metrics(y_test, p)
        rows.append(
            {
                "model": name,
                **pm,
            }
        )
    table = pd.DataFrame(rows)
    ensure_dir(reports_dir)
    table.to_csv(reports_dir / "model_metrics.csv", index=False)

    plots.plot_roc_curves(y_test, {"baseline": p_base, "catboost": p_cat}, reports_dir / "roc.png")
    plots.plot_pr_curves(y_test, {"baseline": p_base, "catboost": p_cat}, reports_dir / "pr.png")
    plots.plot_calibration(
        y_test,
        {"baseline": p_base, "catboost": p_cat},
        reports_dir / "calibration.png",
    )

    # Economic metrics at fixed policy from config
    pol = config["evaluation_policy"]
    policy = PolicyVector(
        tau_replace=float(pol["tau_replace"]),
        safety_stock=float(pol["safety_stock"]),
        order_qty=float(pol["order_qty"]),
    )
    econ_cfg = config["economics"]
    mt = bundle.test["model_type"].values if "model_type" in bundle.test.columns else None
    econ_rows = []
    realized_rows = []
    econ_expected = expected_economic_parameters(econ_cfg)
    for name, p in model_preds:
        em = economic_metrics_fixed_policy(
            y_test, p, policy, econ_cfg, rng, n_scenarios=80, model_types=mt
        )
        econ_rows.append({"model": name, **em})
        if mt is not None:
            realized = realized_policy_profit_from_outcomes(
                y_test,
                p,
                mt,
                policy,
                econ_expected,
            )
            realized_rows.append({"model": name, **realized})
    econ_df = pd.DataFrame(econ_rows)
    econ_df.to_csv(reports_dir / "economic_metrics_fixed_policy.csv", index=False)
    if realized_rows:
        pd.DataFrame(realized_rows).to_csv(
            reports_dir / "economic_metrics_fixed_policy_realized.csv", index=False
        )

    reliability_cfg = config.get("reliability", {})
    n_bootstrap = int(reliability_cfg.get("bootstrap_iterations", 300))
    ci_level = float(reliability_cfg.get("ci_level", 0.95))
    boot_rows: list[dict[str, float | str]] = []
    for name, p in model_preds:
        ci = _bootstrap_metric_ci(
            y_test,
            p,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            rng=rng,
        )
        boot_rows.append({"model": name, **ci})
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(reports_dir / "model_metrics_bootstrap_ci.csv", index=False)
    plots.plot_metric_ci_intervals(
        boot_df,
        reports_dir / "model_metrics_ci.png",
    )

    # CatBoost predictive uncertainty decomposition (instance-wise and summary)
    if pred_cb.get("ensemble") is not None:
        dec_df = _uncertainty_decomposition(pred_cb["ensemble"])
        dec_df.to_csv(reports_dir / "catboost_uncertainty_decomposition.csv", index=False)
        dec_summary = pd.DataFrame(
            [
                {
                    "aleatoric_mean": float(dec_df["aleatoric"].mean()),
                    "epistemic_mean": float(dec_df["epistemic"].mean()),
                    "total_uncertainty_mean": float(dec_df["total_uncertainty"].mean()),
                    "epistemic_share_mean": float(
                        (dec_df["epistemic"] / np.clip(dec_df["total_uncertainty"], 1e-12, None)).mean()
                    ),
                }
            ]
        )
        dec_summary.to_csv(
            reports_dir / "catboost_uncertainty_decomposition_summary.csv",
            index=False,
        )
        plots.plot_uncertainty_decomposition(
            dec_summary.iloc[0].to_dict(), reports_dir / "uncertainty_decomposition.png"
        )

    logger.info("Evaluation complete:\n%s", table.to_string(index=False))
    return table
