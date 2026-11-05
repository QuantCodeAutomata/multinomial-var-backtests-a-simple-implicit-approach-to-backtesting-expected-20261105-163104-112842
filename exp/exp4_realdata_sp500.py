"""Experiment 4: real-data application of multinomial backtests to the
S&P 500 index (Kratz, Lok & McNeil, Sec. 5).

Daily log-returns of the S&P 500 are backtested over the sample period
using four forecasters, each re-estimated every 10 days on a rolling
n2 = 500-day window (Sec. 5):

  * HS:          historical simulation (empirical quantiles of raw losses);
  * GARCH.norm:  GARCH(1,1) fitted with normal innovations;
  * GARCH.t:     GARCH(1,1) fitted with Student-t innovations;
  * GARCH.HS:    GARCH(1,1) (normal QMLE) with empirical quantiles of the
                 standardized residuals.

For every approximately 4-year (about 1000 trading day) block of the
sample, two tests are applied to the accumulated exceptions of that
block (Sec. 5):

  * the multinomial LRT at N = 8 equally spaced levels starting at
    alpha = 0.975 (Eq. 2.1);
  * the one-sided binomial score test of VaR exceptions at the 99% level.

Both yield a p-value and a traffic-light colour (green p >= 0.05,
yellow 0.0001 <= p < 0.05, red p < 0.0001; Sec. 4 / Sec. 5).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src import config
from src.backtest_driver import make_dynamic_block_fn
from src.forecasters import counts_from_forecasts, rolling_backtest_var_forecasts
from src.multinomial_tests import (
    binomial_score_test_one_sided,
    full_alpha_grid,
    lrt_test,
    traffic_light,
)

RESULTS_DIR = "results"
FORECASTERS = ["HS", "GARCH.norm", "GARCH.t", "GARCH.HS"]


def load_losses() -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    """S&P 500 daily losses L_t = -log-return_t over the sample period
    (Sec. 5), via the SPY tracking ETF (see data/data_loader.py)."""
    from data.data_loader import load_sp500_log_returns

    log_returns, meta = load_sp500_log_returns(config.EXP4_START_DATE, config.EXP4_SAMPLE_END, ticker=config.EXP4_TICKER)
    losses = -log_returns.values
    return losses, log_returns.index, meta


def run_realdata_backtest(losses: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rolls every forecaster of `FORECASTERS` forward over the whole
    sample with no look-ahead, then applies the multinomial LRT (N=8) and
    the one-sided binomial score test (alpha=0.99) within each
    approximately `EXP4_PERIOD_LENGTH_YEARS`-year block (Sec. 5).

    Returns a DataFrame with one row per (forecaster, period) with columns
    [forecaster, period_start, period_end, n_days,
     lrt_stat, lrt_p, lrt_colour, binom_z, binom_p, binom_colour].
    """
    n2 = config.EXP4_WINDOW_SIZE
    step = config.EXP4_GARCH_REFIT_STEP
    n_total = len(losses) - n2
    n_total -= n_total % step  # trim to a multiple of `step`, as required by the rolling driver
    losses = losses[: n2 + n_total]
    dates = dates[: n2 + n_total]

    lrt_alpha_grid = full_alpha_grid(config.ALPHA_BASE, config.EXP4_N_LEVELS)
    lrt_interior = lrt_alpha_grid[1:-1]
    binom_alpha = config.EXP4_BINOMIAL_ALPHA
    combined_alpha = np.sort(np.unique(np.concatenate([lrt_interior, [binom_alpha]])))
    lrt_idx = np.searchsorted(combined_alpha, lrt_interior)
    binom_idx = int(np.searchsorted(combined_alpha, binom_alpha))
    assert np.allclose(combined_alpha[lrt_idx], lrt_interior)
    assert np.isclose(combined_alpha[binom_idx], binom_alpha)

    realized = losses[n2:]
    realized_dates = dates[n2:]
    period_len = int(round(config.EXP4_PERIOD_LENGTH_YEARS * 250))  # ~1000 trading days per period (Sec. 5)

    rows = []
    for forecaster in FORECASTERS:
        t0 = time.time()
        block_fn = make_dynamic_block_fn(forecaster)

        def wrapped(window: np.ndarray, future: np.ndarray, horizon: int) -> np.ndarray:
            return block_fn(window, future, horizon, combined_alpha)

        forecasts = rolling_backtest_var_forecasts(losses, n2, step, combined_alpha, wrapped)
        print(f"  forecasts done: {forecaster} ({time.time() - t0:.1f}s)", flush=True)

        n_periods = int(np.ceil(len(realized) / period_len))
        for p in range(n_periods):
            lo, hi = p * period_len, min((p + 1) * period_len, len(realized))
            block_real = realized[lo:hi]
            block_fore = forecasts[lo:hi]

            counts = counts_from_forecasts(block_real, block_fore[:, lrt_idx])
            lrt_stat, lrt_p, _ = lrt_test(counts[None, :], lrt_alpha_grid, kappa=config.KAPPA)

            exceptions = int(np.sum(block_real > block_fore[:, binom_idx]))
            binom_z, binom_p = binomial_score_test_one_sided(np.array([exceptions]), len(block_real), binom_alpha)

            rows.append({
                "forecaster": forecaster,
                "period_start": realized_dates[lo].date().isoformat(),
                "period_end": realized_dates[hi - 1].date().isoformat(),
                "n_days": hi - lo,
                "exceptions_99": exceptions,
                "lrt_stat": float(lrt_stat[0]),
                "lrt_p": float(lrt_p[0]),
                "lrt_colour": str(traffic_light(lrt_p[0], config.TRAFFIC_LIGHT_YELLOW_P, config.TRAFFIC_LIGHT_RED_P)),
                "binom_z": float(binom_z[0]),
                "binom_p": float(binom_p[0]),
                "binom_colour": str(traffic_light(binom_p[0], config.TRAFFIC_LIGHT_YELLOW_P, config.TRAFFIC_LIGHT_RED_P)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="No effect: this experiment always runs the single real S&P 500 "
                              "(SPY) series over the configured sample period.")
    args = parser.parse_args()
    _ = args

    losses, dates, meta = load_losses()
    print(f"Loaded {len(losses)} daily losses from {dates[0].date()} to {dates[-1].date()} "
          f"(ticker_details: {meta.get('ticker_details')})", flush=True)

    t0 = time.time()
    df = run_realdata_backtest(losses, dates)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s; rows={len(df)}", flush=True)

    df.to_csv(f"{RESULTS_DIR}/exp4_realdata_backtest.csv", index=False)
    _plot(df)
    print("Saved results/exp4_realdata_backtest.csv, results/exp4_realdata_sp500.png")


def _plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(FORECASTERS), 1, figsize=(12, 10), sharex=True)
    for row, forecaster in enumerate(FORECASTERS):
        ax = axes[row]
        sub = df[df["forecaster"] == forecaster].reset_index(drop=True)
        x = range(len(sub))
        ax.plot(x, -np.log10(sub["lrt_p"].clip(lower=1e-16)), marker="o", label="Multinomial LRT (N=8)")
        ax.plot(x, -np.log10(sub["binom_p"].clip(lower=1e-16)), marker="s", label="Binomial score (99%)")
        ax.axhline(-np.log10(config.TRAFFIC_LIGHT_YELLOW_P), color="orange", linestyle="--", linewidth=0.8)
        ax.axhline(-np.log10(config.TRAFFIC_LIGHT_RED_P), color="red", linestyle="--", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["period_start"], rotation=45, fontsize=7)
        ax.set_ylabel("-log10(p)")
        ax.set_title(forecaster, fontsize=10)
        if row == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Experiment 4: S&P 500 (SPY) multinomial LRT vs. binomial score test, by 4-year period")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_realdata_sp500.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
