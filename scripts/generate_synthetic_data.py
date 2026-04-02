#!/usr/bin/env python3
"""Generate synthetic SMART-like CSV into data/raw/."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data import generate_synthetic_disk_data
from src.utils import project_root, setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/raw/synthetic_disks.csv")
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    seed = int(cfg["project"]["random_seed"])
    syn = cfg["data"]["synthetic"]
    horizon = int(cfg["project"]["horizon_days"])
    df = generate_synthetic_disk_data(
        n_disks=int(syn["n_disks"]),
        horizon_days=horizon,
        study_span_days=int(syn["study_span_days"]),
        snapshots_per_disk_max=int(syn["snapshots_per_disk_max"]),
        random_seed=seed,
    )
    root = project_root()
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
