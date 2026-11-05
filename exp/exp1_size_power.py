"""Experiment 1: size and power of binomial and multinomial VaR backtests
(Kratz, Lok & McNeil, Sec. 3.1).

For every combination of true distribution G in {normal, t5, t3, skt3},
sample size n and number of levels N, this script estimates:

  * SIZE:  data are drawn from G and the bin edges are the *true* population
    quantiles of G itself (Sec. 3.1.1, "Calculate the true quantiles for N
    levels ... under the correct model"). By the probability-integral
    transform, G(X) ~ Uniform(0,1) for any G, so the vector of bin counts is
    then *exactly* Multinomial(n, p) with p_j = alpha_{j+1} - alpha_j,
    independent of which G was used to generate the data. Sampling this
    multinomial law directly (instead of drawing continuous samples and
    binning them) reproduces the paper's Monte Carlo design exactly, at a
    fraction of the computational cost, and makes the distribution-free
    nature of the size experiment an explicit, checked property rather than
    an implicit numerical coincidence.

  * POWER: data are drawn from G, but the bin edges are the quantiles of the
    benchmark model F = Standard Normal (Sec. 3.1.1, "calculate quantiles
    using a benchmark model F"). The true cell probability of bin j is then
    G.cdf(theta_{j+1}) - G.cdf(theta_j) with theta = F.ppf(alpha), computed
    in closed form from the distributions implemented in src/distributions.py,
    and again sampled exactly via the multinomial law implied by these
    probabilities.

For N = 1 the multinomial design degenerates to the binomial exception test,
evaluated at both alpha = 0.975 and alpha = 0.99 as required by the
methodology; for N >= 2 the Pearson, Nass and LRT multinomial tests are
applied at the single grid starting at alpha = 0.975 (Eq. 2.1).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src import config
from src.distributions import get_distribution
from src.multinomial_tests import (
    binomial_tests,
    bin_counts_from_probabilities,
    full_alpha_grid,
    lrt_test,
    nass_test,
    pearson_test,
)

RESULTS_DIR = "results"


def _multinomial_rejection_rates(
    cell_probs: np.ndarray, alpha_grid: np.ndarray, n: int, n_reps: int, rng: np.random.Generator,
) -> dict[str, float]:
    counts = bin_counts_from_probabilities(n, cell_probs, n_reps, rng)
    rates = {}
    for name, test_fn in (("pearson", pearson_test), ("nass", nass_test), ("lrt", lrt_test)):
        _, _, reject = test_fn(counts, alpha_grid, kappa=config.KAPPA)
        rates[name] = float(reject.mean())
    return rates


def _binomial_rejection_rates(
    exceed_prob: float, n: int, alpha: float, n_reps: int, rng: np.random.Generator,
) -> dict[str, float]:
    exceptions = rng.binomial(n, exceed_prob, size=n_reps)
    results = binomial_tests(exceptions, n, alpha, kappa=config.KAPPA)
    return {name: float(res.reject.mean()) for name, res in results.items()}


def run_size_power_study(n_reps: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (size_df, power_df), each with columns
    [distribution, n, N, test, rejection_rate]."""
    size_rows, power_rows = [], []
    benchmark = get_distribution("normal")

    for dist_name in config.DISTRIBUTIONS_EXP1:
        true_dist = get_distribution(dist_name)
        for n in config.EXP1_SAMPLE_SIZES:
            # --- N = 1: binomial exception test, both 97.5% and 99% levels ---
            for alpha in (config.ALPHA_BASE, config.ALPHA_BINOMIAL_EXTRA):
                size_rates = _binomial_rejection_rates(1 - alpha, n, alpha, n_reps, rng)
                for test, rate in size_rates.items():
                    size_rows.append({"distribution": dist_name, "n": n, "N": 1, "alpha": alpha,
                                       "test": test, "rejection_rate": rate})
                theta_f = benchmark.ppf(np.array([alpha]))[0]
                exceed_prob_power = 1.0 - float(np.atleast_1d(true_dist.cdf(theta_f))[0])
                power_rates = _binomial_rejection_rates(exceed_prob_power, n, alpha, n_reps, rng)
                for test, rate in power_rates.items():
                    power_rows.append({"distribution": dist_name, "n": n, "N": 1, "alpha": alpha,
                                        "test": test, "rejection_rate": rate})

            # --- N >= 2: multinomial Pearson / Nass / LRT tests ---
            for n_levels in config.N_LEVELS:
                if n_levels == 1:
                    continue
                alpha_grid = full_alpha_grid(config.ALPHA_BASE, n_levels)

                size_rates = _multinomial_rejection_rates(np.diff(alpha_grid), alpha_grid, n, n_reps, rng)
                for test, rate in size_rates.items():
                    size_rows.append({"distribution": dist_name, "n": n, "N": n_levels,
                                       "alpha": config.ALPHA_BASE, "test": test, "rejection_rate": rate})

                theta = benchmark.ppf(alpha_grid[1:-1])
                theta_full = np.concatenate([[-np.inf], theta, [np.inf]])
                cdf_vals = np.concatenate([[0.0], true_dist.cdf(theta), [1.0]])
                cell_probs = np.diff(cdf_vals)
                cell_probs = np.clip(cell_probs, 1e-12, None)
                cell_probs = cell_probs / cell_probs.sum()

                power_rates = _multinomial_rejection_rates(cell_probs, alpha_grid, n, n_reps, rng)
                for test, rate in power_rates.items():
                    power_rows.append({"distribution": dist_name, "n": n, "N": n_levels,
                                        "alpha": config.ALPHA_BASE, "test": test, "rejection_rate": rate})

    return pd.DataFrame(size_rows), pd.DataFrame(power_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the paper's full replication count (10,000); default uses the "
                              "same count since this experiment is cheap (closed-form cell probabilities).")
    args = parser.parse_args()

    n_reps = config.EXP1_REPLICATIONS_PAPER if args.full else config.EXP1_REPLICATIONS_RUN
    rng = np.random.default_rng(config.RANDOM_SEED)

    print(f"Running exp_1 size/power study with n_reps={n_reps}, "
          f"distributions={config.DISTRIBUTIONS_EXP1}, sample sizes={config.EXP1_SAMPLE_SIZES}, "
          f"N levels={config.N_LEVELS}")
    t0 = time.time()
    size_df, power_df = run_size_power_study(n_reps, rng)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s; size rows={len(size_df)}, power rows={len(power_df)}")

    size_df.to_csv(f"{RESULTS_DIR}/exp1_size.csv", index=False)
    power_df.to_csv(f"{RESULTS_DIR}/exp1_power.csv", index=False)

    _plot_size_power(size_df, power_df)
    print("Saved results/exp1_size.csv, results/exp1_power.csv, results/exp1_size_power.png")


def _plot_size_power(size_df: pd.DataFrame, power_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    multi_tests = ["pearson", "nass", "lrt"]
    fig, axes = plt.subplots(2, len(config.EXP1_SAMPLE_SIZES), figsize=(20, 8), sharey="row")
    for col, n in enumerate(config.EXP1_SAMPLE_SIZES):
        ax = axes[0, col]
        sub = size_df[(size_df["n"] == n) & (size_df["N"] >= 2) & (size_df["test"].isin(multi_tests))]
        for test in multi_tests:
            agg = sub[sub["test"] == test].groupby("N")["rejection_rate"].mean()
            ax.plot(agg.index, agg.values, marker="o", label=test)
        ax.axhline(config.KAPPA, color="black", linestyle="--", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_title(f"Size, n={n}")
        ax.set_xlabel("N")
        if col == 0:
            ax.set_ylabel("Rejection rate under H0")
            ax.legend()

        ax = axes[1, col]
        sub = power_df[(power_df["n"] == n) & (power_df["N"] >= 2) & (power_df["test"].isin(multi_tests))]
        for test in multi_tests:
            agg = sub[sub["test"] == test].groupby("N")["rejection_rate"].mean()
            ax.plot(agg.index, agg.values, marker="o", label=test)
        ax.set_xscale("log", base=2)
        ax.set_title(f"Power, n={n}")
        ax.set_xlabel("N")
        if col == 0:
            ax.set_ylabel("Rejection rate under H1 (averaged over t5, t3, skt3)")

    fig.suptitle("Size and power of multinomial VaR backtests, averaged over heavy-tailed distributions")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_size_power.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
