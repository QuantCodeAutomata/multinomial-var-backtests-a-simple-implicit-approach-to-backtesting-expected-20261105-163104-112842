"""Experiment 2: static backtesting of VaR models with multinomial tests
(Kratz, Lok & McNeil, Sec. 3.2).

For every true distribution G in {normal, t5, t3, skt3} and rolling
estimation window n2 in {250, 500}, four modellers forecast one-day VaR at
N = 1, 2, 4, 8, 16, 32, 64 equally spaced levels starting at alpha = 0.975
(Eq. 2.1):

  * oracle:   knows G and its exact parameters;
  * good:     fits the correct distributional family (normal/t) by MLE;
  * poor:     always fits a normal distribution;
  * industry: historical simulation (empirical quantiles of the window).

Each modeller is re-estimated every 10 days on the trailing n2-day window
and produces forecasts for the next 10 days (no look-ahead: block b's
forecasts use only the n2 observations strictly preceding it). Rolling
forward over a backtest of length n = 1000 gives 100 re-estimation blocks;
Pearson, Nass and LRT multinomial tests are then applied once to the
accumulated exception counts of the whole backtest, for every N (Sec. 3.2.1).
The whole experiment is replicated to estimate rejection rates.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src import config
from src.backtest_driver import STATIC_MODELLERS, make_static_block_fn, run_rolling_backtest_all_N
from src.distributions import get_distribution
from src.multinomial_tests import full_alpha_grid, lrt_test, nass_test, pearson_test

RESULTS_DIR = "results"


def simulate_iid_losses(dist_name: str, n_total: int, rng: np.random.Generator) -> np.ndarray:
    dist = get_distribution(dist_name)
    return dist.rvs(n_total, rng)


def run_static_backtest_study(n_reps: int, rng: np.random.Generator) -> pd.DataFrame:
    """Returns a DataFrame with columns
    [distribution, n2, modeller, N, test, rejection_rate]."""
    rows = []
    n_levels_list = config.N_LEVELS

    for dist_name in config.DISTRIBUTIONS_EXP1:
        true_dist = get_distribution(dist_name)
        for n2 in config.EXP2_WINDOW_SIZES:
            for modeller in STATIC_MODELLERS:
                block_fn = make_static_block_fn(modeller, true_dist)
                counts_accum = {N: np.zeros((n_reps, N + 1), dtype=int) for N in n_levels_list}
                for r in range(n_reps):
                    losses = simulate_iid_losses(dist_name, n2 + config.EXP2_BACKTEST_LENGTH, rng)
                    counts = run_rolling_backtest_all_N(
                        losses, n2, config.EXP2_STEP, config.ALPHA_BASE, n_levels_list, block_fn
                    )
                    for n_levels in n_levels_list:
                        counts_accum[n_levels][r] = counts[n_levels]

                for n_levels in n_levels_list:
                    alpha_grid = full_alpha_grid(config.ALPHA_BASE, n_levels)
                    for test_name, test_fn in (("pearson", pearson_test), ("nass", nass_test), ("lrt", lrt_test)):
                        _, _, reject = test_fn(counts_accum[n_levels], alpha_grid, kappa=config.KAPPA)
                        rows.append({
                            "distribution": dist_name, "n2": n2, "modeller": modeller,
                            "N": n_levels, "test": test_name, "rejection_rate": float(reject.mean()),
                        })
                print(f"  done: dist={dist_name}, n2={n2}, modeller={modeller}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the paper's full replication count (1000) instead of the "
                              "reduced count used for the default run (see results/RESULTS.md).")
    args = parser.parse_args()

    n_reps = config.EXP2_REPLICATIONS_PAPER if args.full else config.EXP2_REPLICATIONS_RUN
    rng = np.random.default_rng(config.RANDOM_SEED)

    print(f"Running exp_2 static backtest study with n_reps={n_reps}, "
          f"distributions={config.DISTRIBUTIONS_EXP1}, n2={config.EXP2_WINDOW_SIZES}, "
          f"modellers={STATIC_MODELLERS}, N levels={config.N_LEVELS}")
    t0 = time.time()
    df = run_static_backtest_study(n_reps, rng)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s; rows={len(df)}")

    df.to_csv(f"{RESULTS_DIR}/exp2_rejection_rates.csv", index=False)
    _plot(df)
    print("Saved results/exp2_rejection_rates.csv, results/exp2_static_backtest.png")


def _plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modellers = ["oracle", "good", "poor", "industry"]
    n2_values = sorted(df["n2"].unique())
    fig, axes = plt.subplots(len(n2_values), len(modellers), figsize=(20, 9), sharey=True)
    for row, n2 in enumerate(n2_values):
        for col, modeller in enumerate(modellers):
            ax = axes[row, col]
            sub = df[(df["n2"] == n2) & (df["modeller"] == modeller)]
            # average the rejection rate across the four true distributions
            agg = sub.groupby(["test", "N"])["rejection_rate"].mean().reset_index()
            for test_name in ("pearson", "nass", "lrt"):
                t_sub = agg[agg["test"] == test_name].sort_values("N")
                ax.plot(t_sub["N"], t_sub["rejection_rate"], marker="o", label=test_name)
            ax.axhline(config.KAPPA, color="black", linestyle="--", linewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{modeller}, n2={n2}")
            ax.set_xlabel("N")
            if col == 0:
                ax.set_ylabel("Rejection rate")
                ax.legend(fontsize=8)
    fig.suptitle("Experiment 2: static backtest rejection rates by modeller and N "
                 "(averaged over true distributions)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_static_backtest.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
