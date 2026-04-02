"""CatBoost binary classifier with bootstrap ensemble for instance-wise uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CatBoostUncertaintyArtifacts:
    """Ensemble of CatBoost models for mean/variance of failure probability."""

    models: list[CatBoostClassifier]
    feature_columns: list[str]
    cat_features: list[str]
    n_ensemble: int
    random_seed: int


def train_catboost_with_uncertainty(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_features: list[str],
    config: dict[str, Any],
    random_seed: int,
) -> CatBoostUncertaintyArtifacts:
    """
    Train a **bootstrap ensemble** of CatBoost classifiers.

    CatBoost's built-in virtual ensembles / uncertainty APIs vary by version; the
    bootstrap ensemble gives stable per-instance mean and variance of predicted
    failure probability (Bernoulli parameter).
    """
    mc = config["model"]["catboost"]
    n_est = int(mc["bootstrap_ensemble_size"])
    frac = float(mc["bootstrap_sample_frac"])
    rng = np.random.default_rng(random_seed)

    models: list[CatBoostClassifier] = []
    n = len(X_train)
    for k in range(n_est):
        idx = rng.choice(np.arange(n), size=int(n * frac), replace=True)
        Xb = X_train.iloc[idx]
        yb = y_train[idx]

        model = CatBoostClassifier(
            iterations=int(mc["iterations"]),
            depth=int(mc["depth"]),
            learning_rate=float(mc["learning_rate"]),
            loss_function=str(mc["loss_function"]),
            random_seed=random_seed + k,
            thread_count=int(mc.get("thread_count", -1)),
            verbose=bool(mc.get("verbose", False)),
            allow_writing_files=False,
        )
        train_pool = Pool(Xb, yb, cat_features=[c for c in cat_features if c in Xb.columns])
        val_pool = Pool(X_val, y_val, cat_features=[c for c in cat_features if c in X_val.columns])
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)
        models.append(model)
        logger.info("CatBoost ensemble member %d/%d fitted", k + 1, n_est)

    return CatBoostUncertaintyArtifacts(
        models=models,
        feature_columns=list(X_train.columns),
        cat_features=cat_features,
        n_ensemble=n_est,
        random_seed=random_seed,
    )


def predict_with_uncertainty(
    artifacts: CatBoostUncertaintyArtifacts,
    X: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Return mean predicted failure probability and per-row variance across ensemble.

    Returns
    -------
    dict with keys ``mean``, ``variance``, ``ensemble`` (shape n x K).
    """
    cols = artifacts.feature_columns
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing columns for CatBoost: {missing}")

    Xs = X[cols]
    preds = []
    for m in artifacts.models:
        pool = Pool(Xs, cat_features=[c for c in artifacts.cat_features if c in Xs.columns])
        p = m.predict_proba(pool)[:, 1]
        preds.append(p)
    stack = np.vstack(preds)  # K x n
    mean = stack.mean(axis=0)
    var = stack.var(axis=0, ddof=0)
    return {"mean": mean, "variance": var, "ensemble": stack.T}


def save_ensemble(artifacts: CatBoostUncertaintyArtifacts, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(artifacts.models):
        m.save_model(str(directory / f"catboost_member_{i}.cbm"))


def load_ensemble(directory: Path, meta: dict[str, Any]) -> CatBoostUncertaintyArtifacts:
    paths = sorted(directory.glob("catboost_member_*.cbm"))
    models = [CatBoostClassifier().load_model(str(p)) for p in paths]
    return CatBoostUncertaintyArtifacts(
        models=models,
        feature_columns=meta["feature_columns"],
        cat_features=meta["cat_features"],
        n_ensemble=len(models),
        random_seed=int(meta.get("random_seed", 0)),
    )
