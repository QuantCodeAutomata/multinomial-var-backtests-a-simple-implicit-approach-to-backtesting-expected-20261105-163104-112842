"""Experiment 3: dynamic backtesting of VaR models with multinomial tests
(Kratz, Lok & McNeil, Sec. 3.3).

Losses are simulated from a GARCH(1,1) model with Student-t(nu) innovations,
calibrated as stated in Sec. 3.3 (alpha_0 = 2.18e-6, alpha_1 = 0.109,
beta_1 = 0.890, nu = 5.06). Eight forecasters are rolled forward on a
backtest of length n = 1000, re-estimated every 10 steps on a trailing
window of n2 in {500, 1000} observations (no look-ahead):

  * oracle:      true GARCH(1,1)-t parameters known exactly;
  * GARCH.t:     GARCH(1,1) fitted with Student-t innovations (correctly
                 specified dynamics and innovation family);
  * GARCH.HS:    GARCH(1,1) with normal QMLE, VaR from the empirical
                 quantile of standardized residuals ("historical
                 simulation on the residuals");
  * GARCH.EVT:   GARCH(1,1) with normal QMLE, VaR from a Peaks-over-
                 Threshold GPD tail model fitted to standardized residuals;
  * GARCH.norm:  GARCH(1,1) fitted with normal innovations (misspecified
                 innovation distribution, correct dynamics);
  * ARCH.t:      ARCH(1), i.e. misspecified dynamics, correct Student-t
                 innovations;
  * ARCH.norm:   ARCH(1) with normal innovations (misspecified dynamics
                 and innovation distribution);
  * HS:          historical simulation on the raw (unfiltered) losses.

Pearson, Nass and LRT multinomial tests are applied at N = 1, 2, 4, 8, 16,
32, 64 (Eq. 2.1, alpha = 0.975); the whole experiment is replicated to
estimate rejection rates (Sec. 3.3.1).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src import config
from src.backtest_driver import DYNAMIC_FORECASTERS, make_dynamic_block_fn, run_rolling_backtest_all_N
from src.garch_dgp import simulate_garch_t_losses
from src.multinomial_tests import full_alpha_grid, lrt_test, nass_test, pearson_test

RESULTS_DIR = "results"

TRUE_GARCH_PARAMS = (config.GARCH_ALPHA0, config.GARCH_ALPHA1, config.GARCH_BETA1, config.GARCH_NU)


def run_dynamic_backtest_study(n_reps: int, rng: np.random.Generator) -> pd.DataFrame:
    """Returns a DataFrame with columns [n2, forecaster, N, test, rejection_rate]."""
    rows = []
    n_levels_list = config.N_LEVELS

    for n2 in config.EXP3_WINDOW_SIZES:
        for forecaster in DYNAMIC_FORECASTERS:
            block_fn = make_dynamic_block_fn(
                forecaster, true_garch_params=TRUE_GARCH_PARAMS, evt_tail_fraction=config.EVT_TAIL_FRACTION
            )
            counts_accum = {N: np.zeros((n_reps, N + 1), dtype=int) for N in n_levels_list}
            t_start = time.time()
            for r in range(n_reps):
                losses = simulate_garch_t_losses(
                    n2 + config.EXP3_BACKTEST_LENGTH,
                    config.GARCH_ALPHA0, config.GARCH_ALPHA1, config.GARCH_BETA1, config.GARCH_NU, rng,
                )
                counts = run_rolling_backtest_all_N(
                    losses, n2, config.EXP3_STEP, config.ALPHA_BASE, n_levels_list, block_fn
                )
                for n_levels in n_levels_list:
                    counts_accum[n_levels][r] = counts[n_levels]

            for n_levels in n_levels_list:
                alpha_grid = full_alpha_grid(config.ALPHA_BASE, n_levels)
                for test_name, test_fn in (("pearson", pearson_test), ("nass", nass_test), ("lrt", lrt_test)):
                    _, _, reject = test_fn(counts_accum[n_levels], alpha_grid, kappa=config.KAPPA)
                    rows.append({
                        "n2": n2, "forecaster": forecaster, "N": n_levels,
                        "test": test_name, "rejection_rate": float(reject.mean()),
                    })
            print(f"  done: n2={n2}, forecaster={forecaster} ({time.time() - t_start:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the paper's full replication count (500) instead of the "
                              "reduced count used for the default run (see results/RESULTS.md).")
    args = parser.parse_args()

    n_reps = config.EXP3_REPLICATIONS_PAPER if args.full else config.EXP3_REPLICATIONS_RUN
    rng = np.random.default_rng(config.RANDOM_SEED)

    print(f"Running exp_3 dynamic backtest study with n_reps={n_reps}, "
          f"GARCH params={TRUE_GARCH_PARAMS}, n2={config.EXP3_WINDOW_SIZES}, "
          f"forecasters={DYNAMIC_FORECASTERS}, N levels={config.N_LEVELS}", flush=True)
    t0 = time.time()
    df = run_dynamic_backtest_study(n_reps, rng)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s; rows={len(df)}", flush=True)

    df.to_csv(f"{RESULTS_DIR}/exp3_rejection_rates.csv", index=False)
    _plot(df)
    print("Saved results/exp3_rejection_rates.csv, results/exp3_dynamic_backtest.png")


def _plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forecasters = DYNAMIC_FORECASTERS
    n2_values = sorted(df["n2"].unique())
    fig, axes = plt.subplots(len(n2_values), len(forecasters), figsize=(28, 8), sharey=True)
    for row, n2 in enumerate(n2_values):
        for col, forecaster in enumerate(forecasters):
            ax = axes[row, col] if len(n2_values) > 1 else axes[col]
            sub = df[(df["n2"] == n2) & (df["forecaster"] == forecaster)]
            for test_name in ("pearson", "nass", "lrt"):
                t_sub = sub[sub["test"] == test_name].sort_values("N")
                ax.plot(t_sub["N"], t_sub["rejection_rate"], marker="o", label=test_name)
            ax.axhline(config.KAPPA, color="black", linestyle="--", linewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{forecaster}, n2={n2}", fontsize=9)
            ax.set_xlabel("N")
            if col == 0:
                ax.set_ylabel("Rejection rate")
                ax.legend(fontsize=7)
    fig.suptitle("Experiment 3: dynamic (GARCH) backtest rejection rates by forecaster and N")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_dynamic_backtest.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
