#!/usr/bin/bash

git clone https://github.com/linuxhw/SMART.git
poetry install
poetry run python3 prepare.py
./transform_dataset.sh
./transform_dataset_to_csv.sh