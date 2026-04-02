"""Feature encoding and column helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def fit_encode_model_type(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    column: str = "model_type",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """Ordinal-encode ``model_type`` learned on train only."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[[column]])
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train[f"{column}_encoded"] = enc.transform(train[[column]])
    val[f"{column}_encoded"] = enc.transform(val[[column]])
    test[f"{column}_encoded"] = enc.transform(test[[column]])
    return train, val, test, enc


def feature_matrix_for_catboost(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build matrix with ``model_type_encoded`` if present.

    Returns
    -------
    X
        Feature frame.
    cat_features
        Names of categorical columns for CatBoost (subset of columns).
    """
    cols = []
    for c in numeric_cols:
        cols.append(c)
    for c in categorical_cols:
        enc = f"{c}_encoded"
        if enc in df.columns:
            cols.append(enc)
        else:
            cols.append(c)
    X = df[cols].copy()
    for c in X.columns:
        if c.endswith("_encoded"):
            X[c] = X[c].astype(np.int32)
    cat_features = [c for c in cols if c.endswith("_encoded") or c in categorical_cols]
    return X, cat_features
