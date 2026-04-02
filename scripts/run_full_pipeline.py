#!/usr/bin/env python3
"""Run end-to-end pipeline (synthetic or configured CSV)."""

from __future__ import annotations

import argparse

from src.pipeline import run_full_pipeline
from src.utils import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_full_pipeline(args.config)


if __name__ == "__main__":
    main()
