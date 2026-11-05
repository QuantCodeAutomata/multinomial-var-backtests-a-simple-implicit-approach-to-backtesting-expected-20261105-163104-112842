"""Static (iid) VaR forecasters for the static backtest of Sec. 3
("oracle", "good", "poor", "industry" modellers) and the rolling-window
backtest driver shared by the static (exp_2) and dynamic (exp_3, exp_4)
experiments.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.distributions import Distribution
from src.parametric_fits import FIT_FUNCTIONS, quantiles_from_fit

# True family of each simulated distribution (Sec. 3.1), used by the "good"
# modeller to pick which parametric family to fit by MLE.
TRUE_FAMILY = {"normal": "normal", "t5": "t", "t3": "t", "skt3": "skewed_t"}


def static_block_quantiles(
    modeller: str, true_dist: Distribution, window: np.ndarray, horizon: int, alpha_grid: np.ndarray,
) -> np.ndarray:
    """VaR forecasts of a static (iid) modeller, held constant for the
    whole `horizon`-day re-estimation block (Sec. 3.2: "forecasts for the
    next 10 steps").
    """
    if modeller == "oracle":
        q = true_dist.ppf(alpha_grid)
    elif modeller == "good":
        fit = FIT_FUNCTIONS[TRUE_FAMILY[true_dist.name]](window)
        q = quantiles_from_fit(fit, alpha_grid)
    elif modeller == "poor":
        fit = FIT_FUNCTIONS["normal"](window)
        q = quantiles_from_fit(fit, alpha_grid)
    elif modeller == "industry":
        q = np.quantile(window, alpha_grid)
    else:
        raise ValueError(f"Unknown modeller: {modeller}")
    return np.tile(q, (horizon, 1))


def rolling_backtest_counts(
    losses: np.ndarray,
    n2: int,
    step: int,
    alpha_grid_full: np.ndarray,
    block_forecast_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Roll a VaR forecaster forward over `losses`, re-estimating every
    `step` days on the trailing `n2`-day window (Sec. 3.2 / Sec. 4), and
    tabulate the multinomial exception counts O_1..O_{N+1} implied by the
    N interior levels of `alpha_grid_full` (which includes the 0 and 1
    endpoints, so N = len(alpha_grid_full) - 2).

    `block_forecast_fn(window, future_losses, horizon) -> (horizon, N)`
    array of VaR forecasts for the N interior alpha levels, used to bin
    each realized loss with no look-ahead (block t's forecasts use only
    data available strictly before day t within that block, or the
    estimation window preceding the block).

    Returns an integer array of shape (n_blocks, N + 1) with one row of
    bin counts per re-estimation block.
    """
    n = len(losses) - n2
    assert n > 0 and n % step == 0, "backtest length n must be a multiple of `step`"
    n_blocks = n // step
    interior_alpha = alpha_grid_full[1:-1]
    n_bins = len(interior_alpha) + 1
    counts = np.zeros((n_blocks, n_bins), dtype=int)
    for b in range(n_blocks):
        start = n2 + b * step
        window = losses[start - n2:start]
        future = losses[start:start + step]
        var_forecasts = block_forecast_fn(window, future, step)
        assert var_forecasts.shape == (step, len(interior_alpha))
        for t, loss_t in enumerate(future):
            bin_idx = np.searchsorted(var_forecasts[t], loss_t, side="right")
            counts[b, bin_idx] += 1
    return counts


def rolling_backtest_var_forecasts(
    losses: np.ndarray,
    n2: int,
    step: int,
    alpha_grid_interior: np.ndarray,
    block_forecast_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Same rolling, no-look-ahead loop as `rolling_backtest_counts`, but
    returns the full (n, len(alpha_grid_interior)) matrix of per-day VaR
    forecasts instead of already-binned counts. Used once at the finest
    N = 64 grid (Sec. 3.1.1) so that every coarser N's bin counts can be
    derived from the same, single pass of model (re-)fitting -- see
    `subgrid_indices` below for why this is exact, not an approximation.
    """
    n = len(losses) - n2
    assert n > 0 and n % step == 0, "backtest length n must be a multiple of `step`"
    n_blocks = n // step
    out = np.empty((n, len(alpha_grid_interior)))
    for b in range(n_blocks):
        start = n2 + b * step
        window = losses[start - n2:start]
        future = losses[start:start + step]
        out[b * step:(b + 1) * step] = block_forecast_fn(window, future, step)
    return out


def subgrid_indices(n_levels_fine: int, n_levels_coarse: int) -> np.ndarray:
    """0-based positions into the N = `n_levels_fine` interior alpha grid
    that reproduce the N = `n_levels_coarse` interior grid exactly.

    By Eq. (2.1), alpha_j = alpha + (j-1)/N * (1-alpha); for N_coarse
    dividing N_fine, the coarse grid's j-th point (j=1..N_coarse) equals
    the fine grid's point at position 1 + (j-1) * (N_fine / N_coarse), i.e.
    the coarse grid is an exact subsequence of the fine one (both start at
    alpha_1 = alpha regardless of N). This means the finest grid's VaR
    forecasts can be reused for every coarser N without refitting.
    """
    assert n_levels_fine % n_levels_coarse == 0
    ratio = n_levels_fine // n_levels_coarse
    return np.arange(n_levels_coarse) * ratio


def counts_from_forecasts(realized_losses: np.ndarray, var_forecasts: np.ndarray) -> np.ndarray:
    """Bin `realized_losses` (shape (n,)) against a (n, N) matrix of per-day
    VaR forecasts at N interior levels into total exception counts
    O_0, ..., O_N (shape (N+1,)), one-sided as in `rolling_backtest_counts`.
    """
    n_bins = var_forecasts.shape[1] + 1
    bin_idx = np.array([
        np.searchsorted(var_forecasts[t], realized_losses[t], side="right")
        for t in range(len(realized_losses))
    ])
    return np.bincount(bin_idx, minlength=n_bins)
