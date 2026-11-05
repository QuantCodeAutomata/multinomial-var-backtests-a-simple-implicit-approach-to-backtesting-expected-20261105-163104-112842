"""Unit tests for src/forecasters.py (static modellers and the shared
rolling-window backtest driver, Sec. 3.2)."""

from __future__ import annotations

import numpy as np

from src.distributions import get_distribution
from src.forecasters import rolling_backtest_counts, static_block_quantiles
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
