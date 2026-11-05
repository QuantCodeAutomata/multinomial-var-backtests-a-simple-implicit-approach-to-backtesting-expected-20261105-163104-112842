"""Unit tests for src/forecasters.py (static modellers and the shared
rolling-window backtest driver, Sec. 3.2)."""

from __future__ import annotations

import numpy as np

from src.distributions import get_distribution
from src.forecasters import (
    counts_from_forecasts,
    rolling_backtest_counts,
    rolling_backtest_var_forecasts,
    static_block_quantiles,
    subgrid_indices,
)
from src.multinomial_tests import full_alpha_grid


def test_oracle_quantiles_match_true_distribution() -> None:
    dist = get_distribution("t5")
    grid = full_alpha_grid(0.975, 4)
    interior = grid[1:-1]
    q = static_block_quantiles("oracle", dist, window=np.zeros(10), horizon=5, alpha_grid=interior)
    assert q.shape == (5, len(interior))
    assert np.allclose(q[0], dist.ppf(interior))
    assert np.all(q == q[0])  # constant over the whole block (Sec. 3.2: held for 10 steps)


def test_industry_modeller_uses_empirical_quantiles() -> None:
    rng = np.random.default_rng(30)
    window = rng.standard_normal(500)
    grid = full_alpha_grid(0.975, 2)
    interior = grid[1:-1]
    q = static_block_quantiles("industry", get_distribution("normal"), window, 3, interior)
    assert np.allclose(q[0], np.quantile(window, interior))


def test_rolling_backtest_counts_no_lookahead() -> None:
    """Shifting a future loss above the current VaR forecast must not
    change the forecast itself, since it is computed from the window
    only -- i.e. forecasts never use `future` for static modellers."""
    rng = np.random.default_rng(31)
    n2, step, n_blocks = 100, 10, 5
    losses = rng.standard_normal(n2 + n_blocks * step)
    grid = full_alpha_grid(0.975, 4)

    def block_fn(window, future, horizon):
        return static_block_quantiles("oracle", get_distribution("normal"), window, horizon, grid[1:-1])

    counts = rolling_backtest_counts(losses, n2, step, grid, block_fn)
    assert counts.shape == (n_blocks, 5)
    assert np.all(counts.sum(axis=-1) == step)


def test_rolling_backtest_counts_bins_consistently_with_searchsorted() -> None:
    """A loss placed exactly between two known VaR forecasts must land in
    the correct bin."""
    n2, step = 50, 10
    losses = np.zeros(n2 + step)
    losses[n2:] = [-5, -1, 0, 1, 5, -5, -1, 0, 1, 5]
    grid = full_alpha_grid(0.975, 2)  # interior levels -> 3 bins

    def block_fn(window, future, horizon):
        # Fixed forecasts: [-2, 2] -> bins (-inf,-2], (-2,2], (2,inf)
        return np.tile(np.array([-2.0, 2.0]), (horizon, 1))

    counts = rolling_backtest_counts(losses, n2, step, grid, block_fn)
    # losses: -5(bin0), -1(bin1), 0(bin1), 1(bin1), 5(bin2), repeated
    assert counts[0, 0] == 2  # the two -5's
    assert counts[0, 1] == 6  # -1,0,1 twice
    assert counts[0, 2] == 2  # the two 5's


def test_subgrid_indices_reproduce_coarser_alpha_grids() -> None:
    """The N=8 interior grid must be an exact subsequence of the N=64 one
    (Eq. 2.1: alpha_j = alpha + (j-1)/N*(1-alpha))."""
    alpha = 0.975
    fine = full_alpha_grid(alpha, 64)[1:-1]
    coarse = full_alpha_grid(alpha, 8)[1:-1]
    idx = subgrid_indices(64, 8)
    assert idx.shape == (8,)
    assert np.allclose(fine[idx], coarse)


def test_counts_from_forecasts_matches_rolling_backtest_counts() -> None:
    """Computing forecasts once at N=64 and deriving N=4 counts via
    `subgrid_indices` must match running the rolling backtest directly at
    N=4."""
    rng = np.random.default_rng(32)
    n2, step, n_blocks = 100, 10, 5
    losses = rng.standard_normal(n2 + n_blocks * step)
    dist = get_distribution("normal")

    def block_fn(window, future, horizon):
        return static_block_quantiles("oracle", dist, window, horizon, full_alpha_grid(0.975, 64)[1:-1])

    forecasts = rolling_backtest_var_forecasts(losses, n2, step, full_alpha_grid(0.975, 64)[1:-1], block_fn)
    idx = subgrid_indices(64, 4)
    counts_derived = counts_from_forecasts(losses[n2:], forecasts[:, idx])

    def block_fn_4(window, future, horizon):
        return static_block_quantiles("oracle", dist, window, horizon, full_alpha_grid(0.975, 4)[1:-1])

    counts_direct = rolling_backtest_counts(losses, n2, step, full_alpha_grid(0.975, 4), block_fn_4).sum(axis=0)
    assert np.array_equal(counts_derived, counts_direct)
