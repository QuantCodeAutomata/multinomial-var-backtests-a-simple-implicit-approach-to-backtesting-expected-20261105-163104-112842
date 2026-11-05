# Multinomial VaR Backtests

Replication of:

**Kratz, M., Lok, Y. H., & McNeil, A. J. — "Multinomial VaR Backtests: A simple
implicit approach to backtesting expected shortfall"**

This repository reproduces four experiments from the paper: a Monte Carlo
study of the size and power of binomial and multinomial VaR exception tests
(Pearson, Nass, likelihood-ratio), a static backtesting study with oracle,
good, poor and industry (historical-simulation) modellers, a dynamic
GARCH(1,1) backtesting study, and an application of the multinomial LRT and
the binomial score test to 40+ years of S&P 500 daily log-returns.

## Layout

```
src/     shared library: distributions, multinomial/binomial tests, forecasters,
         GARCH data-generating process, EVT tail fit, config.py (paper parameters)
exp/     one script per experiment (exp1_size_power, exp2_static_backtest,
         exp3_dynamic_backtest, exp4_realdata_sp500)
data/    data_loader.py (S&P 500 bars via the QuantX Dataloader REST API)
tests/   pytest unit and market-data tests
results/ CSV/PNG outputs and RESULTS.md with the paper comparison
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY for exp_4
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                         # full suite, requires API keys for exp_4 fixtures
```

## Experiments

```bash
python -m exp.exp1_size_power            # Monte Carlo size/power study (Sec. 3.1)
python -m exp.exp2_static_backtest        # static backtest, reduced replications
python -m exp.exp2_static_backtest --full # paper's 1000 replications (slow)
python -m exp.exp3_dynamic_backtest       # GARCH(1,1) dynamic backtest
python -m exp.exp3_dynamic_backtest --full  # paper's 500 replications
python -m exp.exp4_realdata_sp500         # S&P 500 real-data application (Sec. 5)
```

Each command overwrites its output files directly under `results/`. See
`results/RESULTS.md` for the comparison against the paper's reported figures
and the deviations taken to run the study in the time available.

```bash
bash reproduce.sh   # runs the test suite and every experiment at its executed configuration
```
