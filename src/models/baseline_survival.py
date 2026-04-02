"""Weibull AFT baseline (lifelines) -> P(failure within horizon H)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaselineSurvivalArtifacts:
    """Container for fitted baseline and metadata."""

    fitter: WeibullAFTFitter
    duration_col: str
    event_col: str
    feature_columns: list[str]
    horizon_days: float
    penalizer: float


def train_baseline_survival(
    train_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    feature_columns: list[str],
    horizon_days: float,
    penalizer: float = 0.01,
    l1_ratio: float = 0.0,
) -> BaselineSurvivalArtifacts:
    """
    Fit Weibull AFT on **remaining lifetime** (days) with right-censoring.

    ``duration_col`` is time until failure or censoring from the snapshot;
    ``event_col`` is 1 if failure was observed before censoring.
    """
    df = train_df[[duration_col, event_col, *feature_columns]].copy()
    df = df.dropna()

    wf = WeibullAFTFitter(
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )
    wf.fit(df, duration_col=duration_col, event_col=event_col)
    logger.info("Weibull AFT fitted: %d rows", len(df))

    return BaselineSurvivalArtifacts(
        fitter=wf,
        duration_col=duration_col,
        event_col=event_col,
        feature_columns=feature_columns,
        horizon_days=float(horizon_days),
        penalizer=penalizer,
    )


def predict_failure_prob_in_horizon(
    artifacts: BaselineSurvivalArtifacts,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Predict P(T <= H) where T is remaining lifetime from snapshot, H = horizon.

    Uses the survival function at H: P(fail in (0,H]) = 1 - S(H).
    """
    cols = artifacts.feature_columns
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing feature columns for baseline: {missing}")

    x_df = X[cols].copy()
    # lifelines returns survival at each time point in index
    sf = artifacts.fitter.predict_survival_function(x_df, times=[artifacts.horizon_days])
    # Rows = times, columns = individuals (see lifelines docs)
    s_at_h = sf.iloc[0].values.astype(float)
    p_fail = np.clip(1.0 - s_at_h, 1e-6, 1.0 - 1e-6)
    return p_fail


def artifacts_to_serializable(art: BaselineSurvivalArtifacts) -> dict[str, Any]:
    return {
        "duration_col": art.duration_col,
        "event_col": art.event_col,
        "feature_columns": art.feature_columns,
        "horizon_days": art.horizon_days,
        "penalizer": art.penalizer,
        "params_": art.fitter.params_.to_dict(),
        "summary": art.fitter.summary.to_dict(),
    }
