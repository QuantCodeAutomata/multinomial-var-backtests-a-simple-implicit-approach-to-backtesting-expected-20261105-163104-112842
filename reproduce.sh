#!/usr/bin/env bash
# Reproduces the experiments in this repository at the configuration actually
# executed (see results/RESULTS.md for the full-paper commands and the
# deviations taken to fit the replication's time limit).
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1_size_power
python -m exp.exp2_static_backtest
python -m exp.exp3_dynamic_backtest
python -m exp.exp4_realdata_sp500
