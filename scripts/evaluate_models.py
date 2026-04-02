#!/usr/bin/env python3
"""Evaluate saved or freshly trained models (calls evaluate_models)."""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from src.config import load_config
from src.data import prepare_dataset
from src.evaluation import evaluate_models
from src.features import feature_matrix_for_catboost, fit_encode_model_type
from src.models.baseline_survival import BaselineSurvivalArtifacts
from src.models.catboost_uncertainty import CatBoostUncertaintyArtifacts, load_ensemble
from src.utils import project_root, set_global_seed, setup_logging


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
    X_test, _ = feature_matrix_for_catboost(test, numeric, categorical)
    y_test = test[bundle.target_binary].values

    root = project_root()
    models_dir = root / "outputs" / "models"
    if (models_dir / "baseline_survival.pkl").exists():
        with open(models_dir / "baseline_survival.pkl", "rb") as f:
            baseline: BaselineSurvivalArtifacts = pickle.load(f)

        with open(models_dir / "catboost_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        cat = load_ensemble(models_dir / "catboost_ensemble", meta)
    else:
        from src.models.baseline_survival import train_baseline_survival
        from src.models.catboost_uncertainty import train_catboost_with_uncertainty

        X_train, cat_feats = feature_matrix_for_catboost(train, numeric, categorical)
        X_val, _ = feature_matrix_for_catboost(val, numeric, categorical)
        y_train = train[bundle.target_binary].values
        y_val = val[bundle.target_binary].values
        surv_feats = list(X_train.columns)
        baseline = train_baseline_survival(
            train,
            duration_col=bundle.target_survival_duration,
            event_col=bundle.target_survival_event,
            feature_columns=surv_feats,
            horizon_days=float(raw["project"]["horizon_days"]),
            penalizer=float(raw["model"]["baseline"]["penalizer"]),
            l1_ratio=float(raw["model"]["baseline"]["l1_ratio"]),
        )
        cat = train_catboost_with_uncertainty(
            X_train,
            y_train,
            X_val,
            y_val,
            cat_feats,
            raw,
            random_seed=pcfg.seed,
        )

    evaluate_models(
        bundle,
        baseline,
        cat,
        X_test,
        y_test,
        raw,
        rng,
        pcfg.reports_dir,
    )
    print(f"Reports written to {pcfg.reports_dir}")


if __name__ == "__main__":
    main()
