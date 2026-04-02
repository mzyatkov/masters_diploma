#!/usr/bin/env python3
"""Train baseline + CatBoost and save artifacts to outputs/models/."""

from __future__ import annotations

import argparse
import json

from src.config import load_config
from src.data import prepare_dataset
from src.features import feature_matrix_for_catboost, fit_encode_model_type
from src.models.baseline_survival import train_baseline_survival
from src.models.catboost_uncertainty import save_ensemble, train_catboost_with_uncertainty
from src.utils import ensure_dir, project_root, set_global_seed, setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    pcfg = load_config(args.config)
    set_global_seed(pcfg.seed)
    raw = pcfg.raw
    bundle = prepare_dataset(raw, random_seed=pcfg.seed)

    train, val, test, _ = fit_encode_model_type(bundle.train, bundle.val, bundle.test)
    feat_cfg = raw["features"]
    numeric = list(feat_cfg["numeric_columns"])
    categorical = list(feat_cfg.get("categorical_columns", []))
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

    root = project_root()
    out = ensure_dir(root / "outputs" / "models")
    import pickle

    with open(out / "baseline_survival.pkl", "wb") as f:
        pickle.dump(baseline, f)

    save_ensemble(cat, out / "catboost_ensemble")
    meta = {
        "feature_columns": cat.feature_columns,
        "cat_features": cat.cat_features,
        "random_seed": pcfg.seed,
    }
    with open(out / "catboost_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved models under {out}")


if __name__ == "__main__":
    main()
