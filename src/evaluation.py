"""Model evaluation: ranking metrics, calibration, economic summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data import DatasetBundle
from src.economic_model import PolicyVector, economic_metrics_fixed_policy
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


def evaluate_models(
    bundle: DatasetBundle,
    baseline: BaselineSurvivalArtifacts,
    catboost: CatBoostUncertaintyArtifacts,
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

    rows = []
    for name, p in [("weibull_aft", p_base), ("catboost_ensemble", p_cat)]:
        auc = roc_auc_score(y_test, p)
        pr = average_precision_score(y_test, p)
        brier = brier_score_loss(y_test, p)
        ece, mce = _ece_mce(y_test, p)
        rows.append(
            {
                "model": name,
                "roc_auc": auc,
                "pr_auc": pr,
                "brier": brier,
                "ece": ece,
                "mce": mce,
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
    for name, p in [("weibull_aft", p_base), ("catboost_ensemble", p_cat)]:
        em = economic_metrics_fixed_policy(
            y_test, p, policy, econ_cfg, rng, n_scenarios=80, model_types=mt
        )
        econ_rows.append({"model": name, **em})
    econ_df = pd.DataFrame(econ_rows)
    econ_df.to_csv(reports_dir / "economic_metrics_fixed_policy.csv", index=False)

    logger.info("Evaluation complete:\n%s", table.to_string(index=False))
    return table
