# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of Kratz, M., Lok, Y. H., & McNeil, A. J., "Multinomial VaR
Backtests: A simple implicit approach to backtesting expected shortfall".
Four experiments are implemented:

1. `exp/exp1_size_power.py` — Monte Carlo size and power of binomial (Wald,
   score, LRT) and multinomial (Pearson, Nass, LRT) VaR exception tests
   (paper Sec. 3.1).
2. `exp/exp2_static_backtest.py` — static backtesting with oracle, good, poor
   and industry (historical-simulation) modellers over rolling windows
   (paper Sec. 3.2).
3. `exp/exp3_dynamic_backtest.py` — dynamic backtesting under a GARCH(1,1)
   Student-t data-generating process with eight forecasters, including
   misspecified dynamics and innovation distributions (paper Sec. 3.3).
4. `exp/exp4_realdata_sp500.py` — application of the multinomial LRT and the
   binomial score test to S&P 500 daily log-returns (paper Sec. 5).

## Layout

```
src/     shared library: distributions, binomial/multinomial tests,
         forecasters, GARCH data-generating process, EVT tail fit,
         config.py (all paper-derived parameters as module-level variables)
exp/     one script per experiment, importing only from src/ and data/
data/    data_loader.py — S&P 500 bars through the QuantX Dataloader REST API
tests/   pytest unit tests (synthetic/small fixtures) and market-data tests
results/ CSV/PNG outputs and RESULTS.md with the paper comparison
```

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables with a comment stating their meaning, unit and the
  paper section/equation they come from (or "not specified in the paper").
  No parameter is hard-coded in `exp/` or `src/` modules.
- Market data is loaded only through `data/data_loader.py`, which calls the
  QuantX Dataloader REST API (`https://quantx-api.limex.pro/apis/dataloader`).
  No experiment script fetches data directly.
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from the
  environment via `python-dotenv` and kept only in an uncommitted `.env`
  file (see `.env.example` for the placeholders).
- Deviations from the paper (reduced replication counts, substituted data,
  interpretations of unspecified details) are recorded only in
  `results/RESULTS.md`, never in code comments or config values beyond a
  one-line source note.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY, needed for exp_4
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network, run by default
python -m pytest                          # full suite; the exp_4 fixture needs API keys
```

Tests are organised one file per `src/` module plus `tests/conftest.py`,
which registers the `market_data` marker and provides the cached reduced
S&P 500 slice used by market-data tests.

## Experiments

```bash
python -m exp.exp1_size_power              # Monte Carlo size/power study (Sec. 3.1)
python -m exp.exp2_static_backtest          # static backtest, executed configuration
python -m exp.exp2_static_backtest --full   # paper's 1000 replications (slow)
python -m exp.exp3_dynamic_backtest         # GARCH(1,1) dynamic backtest, executed configuration
python -m exp.exp3_dynamic_backtest --full  # paper's 500 replications (slow)
python -m exp.exp4_realdata_sp500           # S&P 500 real-data application (Sec. 5)
```

Each command is run from the repository root as `python -m exp.<name>` and
overwrites its output files directly under `results/`. After any rerun, the
numbers and verdicts in `results/RESULTS.md` must be checked against the
newly written files and updated by hand — they are not regenerated
automatically.

`reproduce.sh` runs the unit test suite followed by all four experiments at
their executed configuration.
