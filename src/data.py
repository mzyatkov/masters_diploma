"""Data loading, synthetic generation, and leakage-safe splitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetBundle:
    """Train / val / test splits with metadata."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_binary: str
    target_survival_duration: str
    target_survival_event: str
    id_column: str
    time_column: str


def load_csv(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    logger.info("Loading CSV from %s", p)
    df = pd.read_csv(p)
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def generate_synthetic_disk_data(
    n_disks: int = 400,
    horizon_days: int = 90,
    study_span_days: int = 730,
    snapshots_per_disk_max: int = 8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic SMART-like longitudinal snapshots.

    **Assumptions (documented for thesis):**
    - Each disk has a latent Weibull lifetime from install; SMART features are noisy
      functions of age and hazard.
    - ``failure_in_horizon`` indicates failure within the next ``horizon_days`` from
      the snapshot (binary classification target).
    - Survival rows use ``duration_remaining`` (days until failure or end of follow-up)
      and ``event_observed`` (1 if failure observed before censoring).
    """
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []

    model_types = [
        ("budget", 0.0, 1.0),
        ("pro", 0.25, 1.05),
        ("enterprise", 0.5, 1.15),
    ]

    for i in range(n_disks):
        mname, type_shift, cost_scale = model_types[rng.integers(0, 3)]
        # Latent lifetime (days) ~ Weibull; covariates shift scale
        shape = 1.4 + 0.2 * rng.normal()
        scale = np.exp(6.5 + type_shift + 0.3 * rng.normal())
        lifetime = float(rng.weibull(shape) * scale)
        lifetime = max(30.0, lifetime)

        install = rng.integers(0, 120)
        n_snaps = int(rng.integers(1, snapshots_per_disk_max + 1))
        snap_offsets = np.sort(rng.choice(range(30, study_span_days), size=n_snaps, replace=False))

        for off in snap_offsets:
            age_days = float(install + off)
            if age_days >= lifetime:
                continue
            t_event = lifetime - age_days  # remaining life from this snapshot
            cens_horizon = float(study_span_days - off)
            duration_remaining = min(t_event, cens_horizon)
            event_observed = 1 if t_event <= cens_horizon else 0
            failure_in_horizon = 1 if t_event <= horizon_days else 0

            # SMART-like signals (heteroscedastic noise: worse for "rare" enterprise)
            noise_scale = 1.0 + 0.5 * (mname == "enterprise")
            smart_5 = max(0.0, (lifetime - age_days) / lifetime * 100 + rng.normal(0, 5 * noise_scale))
            smart_197 = rng.poisson(0.5 + (age_days / 800) * 3) * (1 if rng.random() < 0.3 else 0)
            smart_198 = rng.poisson(0.2 + (age_days / 1000)) * (1 if rng.random() < 0.2 else 0)
            read_err = max(0.0, rng.exponential(0.02) * (1 + age_days / 500))
            temp_c = float(rng.normal(38 + age_days / 400, 0.5 * noise_scale))

            snapshot_date = datetime(2024, 1, 1) + timedelta(days=int(off))

            rows.append(
                {
                    "disk_id": f"d{i:05d}",
                    "model_type": mname,
                    "age_days": age_days,
                    "smart_5_reallocated": smart_5,
                    "smart_197_current_pending": float(smart_197),
                    "smart_198_offline_uncorrectable": float(smart_198),
                    "read_errors_rate": read_err,
                    "temperature_c": temp_c,
                    "snapshot_date": snapshot_date,
                    "failure_in_horizon": int(failure_in_horizon),
                    "duration_remaining": float(duration_remaining),
                    "event_observed": int(event_observed),
                    "cost_scale": float(cost_scale),
                    "latent_lifetime": lifetime,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Synthetic generation produced empty frame; increase study_span_days.")
    logger.info("Synthetic dataset: %d rows, %d unique disks", len(df), df["disk_id"].nunique())
    return df


def _disk_level_split(
    df: pd.DataFrame,
    id_column: str,
    time_column: str,
    train_frac: float,
    val_frac: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split by disk: order disks by **last snapshot time** (ascending), then
    contiguous segments [train | val | test].

    Avoids the same disk appearing in multiple splits. ``random_seed`` is kept for API
    stability; ordering is deterministic given data.
    """
    _ = random_seed  # reserved if we add stratified variants
    last_time = df.groupby(id_column)[time_column].max().sort_values()
    disk_ids = last_time.index.tolist()
    n = len(disk_ids)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_ids = set(disk_ids[:n_train])
    val_ids = set(disk_ids[n_train : n_train + n_val])
    test_ids = set(disk_ids[n_train + n_val :])

    train = df[df[id_column].isin(train_ids)].copy()
    val = df[df[id_column].isin(val_ids)].copy()
    test = df[df[id_column].isin(test_ids)].copy()
    return train, val, test


def prepare_dataset(
    config: dict[str, Any],
    random_seed: int | None = None,
) -> DatasetBundle:
    """
    Load or synthesize data, build feature list, and split without disk leakage.

    Parameters
    ----------
    config
        Full YAML config dict (``load_config``).
    random_seed
        Overrides ``project.random_seed`` if set.
    """
    seed = int(random_seed if random_seed is not None else config["project"]["random_seed"])
    data_cfg = config["data"]
    feat_cfg = config["features"]
    split_cfg = data_cfg["split"]
    horizon = int(config["project"]["horizon_days"])

    path = data_cfg.get("csv_path")
    if path:
        df = load_csv(path)
    else:
        syn = data_cfg["synthetic"]
        df = generate_synthetic_disk_data(
            n_disks=int(syn["n_disks"]),
            horizon_days=horizon,
            study_span_days=int(syn["study_span_days"]),
            snapshots_per_disk_max=int(syn["snapshots_per_disk_max"]),
            random_seed=seed,
        )

    id_col = feat_cfg["id_column"]
    time_col = feat_cfg["time_column"]
    numeric = list(feat_cfg["numeric_columns"])
    categorical = list(feat_cfg.get("categorical_columns", []))

    for c in numeric + categorical + [id_col, time_col]:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in dataset.")

    feature_columns = numeric + categorical

    train, val, test = _disk_level_split(
        df,
        id_column=id_col,
        time_column=time_col,
        train_frac=float(split_cfg["train_frac"]),
        val_frac=float(split_cfg["val_frac"]),
        random_seed=seed,
    )

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d (disk-level)",
        train[id_col].nunique(),
        val[id_col].nunique(),
        test[id_col].nunique(),
    )

    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        feature_columns=feature_columns,
        target_binary="failure_in_horizon",
        target_survival_duration="duration_remaining",
        target_survival_event="event_observed",
        id_column=id_col,
        time_column=time_col,
    )
