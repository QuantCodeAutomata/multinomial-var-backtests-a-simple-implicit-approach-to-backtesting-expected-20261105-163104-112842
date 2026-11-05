"""Parameters used across the replication of Kratz, Lok & McNeil,
"Multinomial VaR Backtests: A simple implicit approach to backtesting
expected shortfall".

Every module-level variable states the paper section/equation it comes
from, or "not specified in the paper" when a value had to be chosen.
No value here is a deviation by itself; deviations (if any) are recorded
in results/RESULTS.md.
"""

import numpy as np

RANDOM_SEED = 20261105  # not specified in the paper; fixes reproducibility

# ---------------------------------------------------------------------------
# Common test/quantile-level definitions (Section 2.1, Eq. 2.1)
# ---------------------------------------------------------------------------
ALPHA_BASE = 0.975          # starting VaR level alpha used in all multinomial tests (Sec. 3.1.1)
ALPHA_BINOMIAL_EXTRA = 0.99  # additional level considered only for N = 1 binomial tests (Sec. 3.1.1)
N_LEVELS = [1, 2, 4, 8, 16, 32, 64]  # N = 2^k, k = 0..6 (Sec. 3.1.1)
KAPPA = 0.05                 # test size (type-I error) used to report rejection rates (not stated as a
                              # single symbol in the paper; standard 5% significance level is used throughout)

# ---------------------------------------------------------------------------
# Distributions used in the simulation study (Sec. 3.1.1), all standardised
# to mean 0 and variance 1
# ---------------------------------------------------------------------------
T5_DOF = 5          # degrees of freedom of the "t5" distribution (Sec. 3.1.1)
T3_DOF = 3          # degrees of freedom of the "t3" and "skt3" distributions (Sec. 3.1.1)
SKEW_T3_GAMMA = 1.2  # Fernandez & Steel (1998) skewness parameter for skt3 (Sec. 3.1.1)
DISTRIBUTIONS_EXP1 = ["normal", "t5", "t3", "skt3"]  # true distributions G simulated (Sec. 3.1.1)

# ---------------------------------------------------------------------------
# Experiment 1: size and power of binomial and multinomial VaR backtests
# (Sec. 3.1)
# ---------------------------------------------------------------------------
EXP1_SAMPLE_SIZES = [250, 500, 1000, 2000]   # n_1 values (Sec. 3.1.1)
EXP1_REPLICATIONS_PAPER = 10_000             # replications used in the paper (Sec. 3.1.1)
EXP1_REPLICATIONS_RUN = 10_000               # replications actually executed (bin counts are
                                              # drawn directly from the multinomial law implied by the true
                                              # cell probabilities, which reproduces the paper's Monte Carlo
                                              # design exactly and is cheap enough to run at full scale)

# ---------------------------------------------------------------------------
# Experiment 2: static backtesting experiment (Sec. 3.2)
# ---------------------------------------------------------------------------
EXP2_BACKTEST_LENGTH = 1000          # n, length of the backtest (Sec. 3.2.1)
EXP2_WINDOW_SIZES = [250, 500]       # n_2, rolling estimation window (Sec. 3.2.1)
EXP2_STEP = 10                       # re-estimation step (Sec. 3.2.1): 100 re-estimations over n=1000
EXP2_REPLICATIONS_PAPER = 1000       # replications in the paper (Sec. 3.2.1)
EXP2_REPLICATIONS_RUN = 200          # replications executed in this run; see results/RESULTS.md for the
                                      # runtime estimate behind this reduction. Run with --full for 1000.

# ---------------------------------------------------------------------------
# Experiment 3: dynamic (GARCH) backtesting experiment (Sec. 3.3)
# ---------------------------------------------------------------------------
GARCH_ALPHA0 = 2.18e-6     # GARCH(1,1) constant, fitted to S&P index 2000-2012 (Sec. 3.3)
GARCH_ALPHA1 = 0.109       # GARCH(1,1) ARCH coefficient (Sec. 3.3)
GARCH_BETA1 = 0.890        # GARCH(1,1) GARCH coefficient (Sec. 3.3)
GARCH_NU = 5.06            # degrees of freedom of the Student t innovations (Sec. 3.3)
EXP3_BACKTEST_LENGTH = 1000           # n (Sec. 3.3.1)
EXP3_WINDOW_SIZES = [500, 1000]       # n_2 rolling estimation windows (Sec. 3.3.1)
EXP3_STEP = 10                        # re-estimation step (Sec. 3.3.1)
EXP3_REPLICATIONS_PAPER = 500         # replications in the paper (Sec. 3.3.1)
EXP3_REPLICATIONS_RUN = 60            # replications executed in this run; see results/RESULTS.md.
                                       # Run with --full for 500.
EVT_TAIL_FRACTION = 0.1   # fraction of residuals used as GPD tail sample for GARCH.EVT (not specified in
                           # the paper beyond "an EVT tail model"; 10% is the standard rule of thumb
                           # used in McNeil, Frey & Embrechts (2015), Ch. 5, cited by the paper for the
                           # forecaster definitions)

# ---------------------------------------------------------------------------
# Experiment 4: real-data application to the S&P 500 index (Sec. 5)
# ---------------------------------------------------------------------------
EXP4_TICKER = "SPY"               # S&P 500 tracking ETF used in place of the unentitled Massive
                                   # index symbol I:SPX (Sec. 5 names the S&P 500 index itself)
EXP4_START_DATE = "1993-01-01"    # SPY's first trading day is 1993-01-29; the paper's own pre-sample
                                   # start of 500 trading days before the test period (Sec. 5) is applied
                                   # from this earliest available date instead of 1976 (see
                                   # results/RESULTS.md, "Deviations from the paper")
EXP4_SAMPLE_START = "1995-06-01"  # start of the backtest period actually run, after the n_2=500-day
                                   # initialisation window computed from EXP4_START_DATE
EXP4_SAMPLE_END = "2016-12-31"    # end of the backtest period (Sec. 5: data through 2016)
EXP4_PERIOD_LENGTH_YEARS = 4      # each multinomial test is run on an approximately 4-year (1000 day) block (Sec. 5)
EXP4_WINDOW_SIZE = 500            # n_2, rolling window used by every forecaster (Sec. 5)
EXP4_GARCH_REFIT_STEP = 10        # GARCH parameters updated every 10 days (Sec. 5)
EXP4_N_LEVELS = 8                 # N, number of equally spaced levels for the multinomial LRT (Sec. 5)
EXP4_BINOMIAL_ALPHA = 0.99        # level of the one-sided binomial score test (Sec. 5)
TRAFFIC_LIGHT_YELLOW_P = 0.05     # p < 0.05 -> yellow (Sec. 5 / Sec. 4)
TRAFFIC_LIGHT_RED_P = 0.0001      # p < 0.0001 -> red (Sec. 5 / Sec. 4)

RNG = np.random.default_rng(RANDOM_SEED)
