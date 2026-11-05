"""Shared rolling-window VaR backtest driver used by the static (Sec. 3.2),
dynamic (Sec. 3.3) and real-data (Sec. 5) backtesting experiments.

Fits each re-estimation block only once, at the finest N = 64 interior
alpha grid (Sec. 3.1.1, Eq. 2.1), and derives every coarser N's bin counts
from that single set of forecasts via `forecasters.subgrid_indices`
(exact, since coarser grids are nested subsequences of the finest one).
This reproduces exactly the methodology described for each experiment
("apply Pearson, Nass and LRT tests for N = 1, ..., 64") while fitting the
model only once per block instead of once per (block, N) pair.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from src.distributions import Distribution, get_distribution
from src.evt import evt_quantile
from src.forecasters import (
    counts_from_forecasts,
    rolling_backtest_var_forecasts,
    static_block_quantiles,
    subgrid_indices,
)
from src.garch_dgp import (
    fit_arch1,
    fit_garch_norm,
    fit_garch_t,
    garch_block_quantiles,
    oracle_block_quantiles,
)
from src.multinomial_tests import full_alpha_grid

BlockForecastFn = Callable[[np.ndarray, np.ndarray, int], np.ndarray]

# Static (iid) modellers of the Sec. 3.2 experiment.
STATIC_MODELLERS = ["oracle", "good", "poor", "industry"]

# Dynamic (GARCH-based) forecasters of the Sec. 3.3 / Sec. 5 experiments.
DYNAMIC_FORECASTERS = [
    "oracle", "GARCH.t", "GARCH.HS", "GARCH.EVT", "GARCH.norm", "ARCH.t", "ARCH.norm", "HS",
]


def make_static_block_fn(modeller: str, true_dist: Distribution) -> BlockForecastFn:
    """Block-forecast closure for a static modeller (Sec. 3.2.1)."""

    def block_fn(window: np.ndarray, future: np.ndarray, horizon: int, _alpha: np.ndarray) -> np.ndarray:
        return static_block_quantiles(modeller, true_dist, window, horizon, _alpha)

    return block_fn


def make_dynamic_block_fn(
    forecaster: str,
    true_garch_params: Optional[tuple[float, float, float, float]] = None,
    evt_tail_fraction: float = 0.1,
) -> BlockForecastFn:
    """Block-forecast closure for a dynamic (GARCH-based) forecaster
    (Sec. 3.3.1: Oracle, GARCH.t, GARCH.HS, GARCH.EVT, GARCH.norm, ARCH.t,
    ARCH.norm, HS)."""

    def block_fn(window: np.ndarray, future: np.ndarray, horizon: int, alpha: np.ndarray) -> np.ndarray:
        if forecaster == "HS":
            q = np.quantile(window, alpha)
            return np.tile(q, (horizon, 1))
        if forecaster == "oracle":
            omega, alpha1, beta1, dof = true_garch_params
            return oracle_block_quantiles(omega, alpha1, beta1, dof, window, future, alpha)
        if forecaster == "GARCH.t":
            fit = fit_garch_t(window)
            return garch_block_quantiles(fit, future, alpha, "t")
        if forecaster == "GARCH.norm":
            fit = fit_garch_norm(window)
            return garch_block_quantiles(fit, future, alpha, "normal")
        if forecaster == "GARCH.HS":
            fit = fit_garch_norm(window)
            return garch_block_quantiles(fit, future, alpha, "hs")
        if forecaster == "GARCH.EVT":
            fit = fit_garch_norm(window)
            return garch_block_quantiles(fit, future, alpha, "evt", evt_tail_fraction)
        if forecaster == "ARCH.t":
            fit = fit_arch1(window, "t")
            return garch_block_quantiles(fit, future, alpha, "t")
        if forecaster == "ARCH.norm":
            fit = fit_arch1(window, "normal")
            return garch_block_quantiles(fit, future, alpha, "normal")
        raise ValueError(f"Unknown dynamic forecaster: {forecaster}")

    return block_fn


def run_rolling_backtest_all_N(
    losses: np.ndarray,
    n2: int,
    step: int,
    alpha_base: float,
    n_levels_list: list[int],
    block_fn: BlockForecastFn,
) -> dict[int, np.ndarray]:
    """Run one rolling backtest over `losses` and return, for every N in
    `n_levels_list`, the total exception-count vector O_0..O_N (Sec. 2.1)
    accumulated over the whole backtest window.
    """
    n_fine = max(n_levels_list)
    alpha_grid_fine = full_alpha_grid(alpha_base, n_fine)
    alpha_interior_fine = alpha_grid_fine[1:-1]

    def wrapped(window: np.ndarray, future: np.ndarray, horizon: int) -> np.ndarray:
        return block_fn(window, future, horizon, alpha_interior_fine)

    forecasts = rolling_backtest_var_forecasts(losses, n2, step, alpha_interior_fine, wrapped)
    realized = losses[n2:]
    counts = {}
    for n_levels in n_levels_list:
        idx = subgrid_indices(n_fine, n_levels)
        counts[n_levels] = counts_from_forecasts(realized, forecasts[:, idx])
    return counts


def simulate_iid_losses(dist_name: str, n_total: int, rng: np.random.Generator) -> np.ndarray:
    """n_total iid losses from a standardized distribution named in
    `src.distributions.DISTRIBUTIONS` (Sec. 3.1.1)."""
    dist = get_distribution(dist_name)
    return dist.rvs(n_total, rng)
